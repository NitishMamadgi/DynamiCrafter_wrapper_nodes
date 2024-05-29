
import os
from omegaconf import OmegaConf  # YAML/JSON configuration library
import torch  # PyTorch library
import torch.nn.functional as F  # Activation functions
from .scripts.evaluation.funcs import load_model_checkpoint, get_latent_z  # Load model checkpoint and get latent z
from .utils.utils import instantiate_from_config  # Instantiate from config
from einops import repeat  # Tensor operations
import folder_paths  # Folder paths
import comfy.model_management as mm  # Model management
import comfy.utils  # Utils
from contextlib import nullcontext  # Context manager
from .lvdm.models.samplers.ddim import DDIMSampler  # DDIM sampler


def split_and_trim(input_string):
    array = input_string.split('|')

    trimmed_array = [element.strip() for element in array]
    
    return trimmed_array

def convert_dtype(dtype_str):
    if dtype_str == 'fp32': 
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError
    
script_directory = os.path.dirname(os.path.abspath(__file__))

class DynamiCrafterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto',
                    ], {
                        "default": 'auto'
                    }),
            "fp8_unet": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("DCMODEL",)
    RETURN_NAMES = ("DynCraft_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DynamiCrafterWrapper_qsn"

    def loadmodel(self, dtype, ckpt_name, fp8_unet=False):
        # Empty the soft cache
        mm.soft_empty_cache()

        # Set the custom configuration
        custom_config = {
            'dtype': dtype,
            'ckpt_name': ckpt_name,
        }

        # Load the model if it is not already loaded or if the configuration has changed
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.current_config = custom_config

            # Get the path of the checkpoint file
            model_path = folder_paths.get_full_path("checkpoints", ckpt_name)

            # Get the base name of the checkpoint file
            ckpt_base_name = os.path.basename(ckpt_name)
            base_name, _ = os.path.splitext(ckpt_base_name)

            # Load the configuration file based on the base name
            if 'interp' in base_name and '512' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_512_interp_v1.yaml")
            elif '1024' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_1024_v1.yaml")
            elif '512' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_512_v1.yaml")
            elif '256' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_256_v1.yaml")
            else:
                print(f"No matching config for model: {ckpt_name}")
            config = OmegaConf.load(config_file)  # something like a dict ==> omegaconf.dictconfig.DictConfig

            # Load the model configuration
            model_config = config.pop("model", OmegaConf.create())
            model_config['params']['unet_config']['params']['use_checkpoint']=False   
            self.model = instantiate_from_config(model_config)

            # Load the model checkpoint  --> A class object
            self.model = load_model_checkpoint(self.model, model_path)

            # Set the model to evaluation mode
            self.model.eval()

            # Set the data type of the model
            if dtype == "auto":
                try:
                    if mm.should_use_fp16():
                        self.model.to(convert_dtype('fp16'))
                    elif mm.should_use_bf16():
                        self.model.to(convert_dtype('bf16'))
                    else:
                        self.model.to(convert_dtype('fp32'))
                except:
                    raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
            else:
                self.model.to(convert_dtype(dtype))

            # Set the data type of the U-Net to fp8 if fp8_unet is True
            if fp8_unet:
                self.model.model.diffusion_model = self.model.model.diffusion_model.to(torch.float8_e4m3fn)

            # Print the data type of the model
            print(f"Model using dtype: {self.model.dtype}")

            # Print the model
            print(self.model)

        # Return the loaded model
        return (self.model,)


# ..............................................................................................................................
#...............................................................................................................................

device = mm.get_torch_device()
def device_setup(model):
    model.to(device)
    dtype = model.dtype
    return dtype, (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)

class Dynamic_dtype:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "vae_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

            }
        }
    
    RETURN_TYPES = ("DCMODEL",)
    RETURN_NAMES = ("dmodel",)
    FUNCTION = "convert_dtype"
    CATEGORY = "DynamiCrafterWrapper_qsn"

    def convert_dtype(self, model, vae_dtype, seed):

        mm.unload_all_models()
        mm.soft_empty_cache()

        torch.manual_seed(seed)
        
        if vae_dtype == "auto":
            try:
                if mm.should_use_bf16():
                    model.first_stage_model.to(convert_dtype('bf16'))
                else:
                    model.first_stage_model.to(convert_dtype('fp32'))
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
        else:
            model.first_stage_model.to(convert_dtype(vae_dtype))
        print(f"VAE using dtype: {model.first_stage_model.dtype}")

        self.model = model

        return (self.model,)

# ..............................................................................................................................
#...............................................................................................................................

class Dynamic_encode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "image": ("IMAGE",),
            "frames": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
            }, 
            "optional": {
                "image2": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LIST")
    RETURN_NAMES = ("latents","noise_shape")
    FUNCTION = "vae_encode"
    CATEGORY = "DynamiCrafterWrapper_qsn"

    def vae_encode(self, model, image, frames, image2=None):
        self.model = model
        dtype, autocast = device_setup(self.model)

        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast else nullcontext():
            
            image = image * 2 - 1
            image = image.permute(0, 3, 1, 2).to(dtype).to(device)

            B, C, H, W = image.shape
            orig_H, orig_W = H, W
            if W % 64 != 0:
                W = W - (W % 64)
            if H % 64 != 0:
                H = H - (H % 64)
            if orig_H % 64 != 0 or orig_W % 64 != 0:
                image = F.interpolate(image, size=(H, W), mode="bicubic")

            B, C, H, W = image.shape
            noise_shape = [B, self.model.model.diffusion_model.out_channels, frames, H // 8, W // 8]
    
            self.model.first_stage_model.to(device)

            z = get_latent_z(self.model, image.unsqueeze(2)) #bc,1,hw

            if image2 is not None:
                image2 = image2 * 2 - 1
                image2 = image2.permute(0, 3, 1, 2).to(dtype).to(device)
                if image2.shape != image.shape:
                    image2 = F.interpolate(image2, size=(H, W), mode="bicubic")
                z2 = get_latent_z(self.model, image2.unsqueeze(2)) #bc,1,hw
                img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)
                img_tensor_repeat = torch.zeros_like(img_tensor_repeat)
                img_tensor_repeat[:,:,:1,:,:] = z
                img_tensor_repeat[:,:,-1:,:,:] = z2
            else:
                img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

            self.model.first_stage_model.to('cpu')

            return (img_tensor_repeat, noise_shape)
    

# ..............................................................................................................................
#...............................................................................................................................

class Dynamic_Image_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("img_emb",)
    FUNCTION = "image_conditioning"
    CATEGORY = "DynamiCrafterWrapper_qsn"

    def image_conditioning(self, model, image):
        self.model = model
        dtype, autocast = device_setup(self.model)

        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast else nullcontext():
            image = image * 2 - 1
            image = image.permute(0, 3, 1, 2).to(dtype).to(device)

            B, C, H, W = image.shape
            orig_H, orig_W = H, W
            if W % 64 != 0:
                W = W - (W % 64)
            if H % 64 != 0:
                H = H - (H % 64)
            if orig_H % 64 != 0 or orig_W % 64 != 0:
                image = F.interpolate(image, size=(H, W), mode="bicubic")
    
            self.model.cond_stage_model.to(device)
            self.model.embedder.to(device)
            self.model.image_proj_model.to(device)

            cond_images = self.model.embedder(image)
            img_emb = self.model.image_proj_model(cond_images)
            del cond_images

            return (img_emb,)

# ..............................................................................................................................
#...............................................................................................................................
class Dynamic_Text_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "prompt": ("STRING", {"multiline": True, "default": "",}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("text_emb",)
    FUNCTION = "text_conditioning"
    CATEGORY = "DynamiCrafterWrapper_qsn"

    def text_conditioning(self, model, prompt):
        self.model = model
        dtype, autocast = device_setup(self.model)

        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast else nullcontext():
            self.model.cond_stage_model.to(device)

            text_emb = self.model.get_learned_conditioning([prompt])
            return (text_emb,)
        
# ..............................................................................................................................
#...............................................................................................................................

class Text_Image_Conditioning_Combine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "img_emb": ("CONDITIONING",),
            "text_emb": ("CONDITIONING",),
            }
        }    
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("imtext_emb",)
    FUNCTION = "conditioning_combine"
    CATEGORY = "DynamiCrafterWrapper_qsn"

    def conditioning_combine(self, img_emb, text_emb):
        imtext_emb = torch.cat((text_emb, img_emb), dim=1)
        del img_emb, text_emb
        return (imtext_emb,)


# ..............................................................................................................................
#...............................................................................................................................

class Dynamic_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "imtext_cond": ("CONDITIONING",),
            "img_tensor_repeat": ("LATENT",),
            "noise_shape": ("LIST",),
            "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "fs": ("INT", {"default": 10, "min": 2, "max": 100, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "frame_window_size": ("INT", {"default": 16, "min": 1, "max": 200, "step": 1}),
                "frame_window_stride": ("INT", {"default": 4, "min": 1, "max": 200, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "DynamiCrafterWrapper_qsn"

    def process(self,model, imtext_cond,img_tensor_repeat, noise_shape, cfg, steps, eta, fs, frame_window_size=16, frame_window_stride=4, mask=None ):
        self.model = model
        dtype, autocast = device_setup(self.model)

        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast else nullcontext():

            cond = {"c_crossattn": [imtext_cond], "c_concat": [img_tensor_repeat]}

            if cfg != 1.0:
                if imtext_cond.shape[1] == 333:
                    uc_txt = self.model.get_learned_conditioning([""])
                    ## process image embedding token
                    if hasattr(self.model, 'embedder'):
                        uc_img = torch.zeros(noise_shape[0],3,224,224).to(self.model.device)
                        ## img: b c h w >> b l c
                        uc_img = self.model.embedder(uc_img)
                        uc_img = self.model.image_proj_model(uc_img)
                        uc_emb = torch.cat([uc_txt, uc_img], dim=1)
                    else:
                        uc_emb = uc_txt
                    if isinstance(cond, dict):
                        uc = {key:cond[key] for key in cond.keys()}
                        uc.update({'c_crossattn': [uc_emb]})

                elif imtext_cond.shape[1] == 77:
                    uc_emb = self.model.get_learned_conditioning([""])
                    if isinstance(cond, dict):
                        uc = {key:cond[key] for key in cond.keys()}
                        uc.update({'c_crossattn': [uc_emb]})

                else:
                    if hasattr(self.model, 'embedder'):
                        uc_img = torch.zeros(noise_shape[0],3,224,224).to(self.model.device)
                        ## img: b c h w >> b l c
                        uc_img = self.model.embedder(uc_img)
                        uc_img = self.model.image_proj_model(uc_img)
                        uc_emb = uc_img
                    if isinstance(cond, dict):
                        uc = {key:cond[key] for key in cond.keys()}
                        uc.update({'c_crossattn': [uc_emb]})
            else:
                uc = None

            self.model.cond_stage_model.to('cpu')
            self.model.embedder.to('cpu')
            self.model.image_proj_model.to('cpu')

            fs = torch.tensor([fs], dtype=torch.long, device=self.model.device)
        
            if noise_shape[-1] == 32:
                timestep_spacing = "uniform"
                guidance_rescale = 0.0
            else:
                timestep_spacing = "uniform_trailing"
                guidance_rescale = 0.7

            if mask is not None:     
                mask = mask.to(dtype).to(device)
                mask = F.interpolate(mask.unsqueeze(0), size=(H // 8, W // 8), mode="nearest").squeeze(0)
                mask = (1 - mask)
                mask = mask.unsqueeze(1)
                B, C, H, W = mask.shape
                if B < noise_shape[2]:
                    mask = mask.unsqueeze(2)
                    mask = mask.expand(-1, -1, noise_shape[2], -1, -1)
                else:
                    mask = mask.unsqueeze(0)
                    mask = mask.permute(0, 2, 1, 3, 4) 
                mask = torch.where(mask < 1.0, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype))
            
            #inference
            ddim_sampler = DDIMSampler(self.model)
            samples, _ = ddim_sampler.sample(S=steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=True,
                                            unconditional_guidance_scale=cfg,
                                            unconditional_conditioning=uc,
                                            eta=eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=None,
                                            x_T=None,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            clean_cond=True,
                                            mask=mask,
                                            x0=img_tensor_repeat.clone() if mask is not None else None,
                                            frame_window_size = frame_window_size,
                                            frame_window_stride = frame_window_stride
                                            )
            
            assert not torch.isnan(samples).any().item(), "Resulting tensor containts NaNs. I'm unsure why this happens, changing step count and/or image dimensions might help."
            
            return (samples,)

# ..............................................................................................................................
#...............................................................................................................................

class sample_decode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "samples": ("LATENT",),
            "keep_model_loaded" : ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "vae_decode"
    CATEGORY = "DynamiCrafterWrapper_qsn"
        
    def vae_decode(self, model, samples, keep_model_loaded):
        self.model = model
        dtype, autocast = device_setup(self.model)

        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast else nullcontext():

            self.model.first_stage_model.to(device)
            decoded_images = self.model.decode_first_stage(samples) #b c t h w
            self.model.first_stage_model.to('cpu')
            del samples

            video = decoded_images.detach().cpu()
            video = torch.clamp(video.float(), -1., 1.)
            video = (video + 1.0) / 2.0
            video = video.squeeze(0).permute(1, 2, 3, 0)
            del decoded_images
            
            if not keep_model_loaded:
                self.model.to('cpu')
                mm.soft_empty_cache()

            return (video,)

            
NODE_CLASS_MAPPINGS = {
    "DynamiCrafterModelLoader": DynamiCrafterModelLoader,
    "Dynamic_dtype": Dynamic_dtype,
    "Dynamic_encode": Dynamic_encode,
    "Dynamic_Image_Conditioning": Dynamic_Image_Conditioning,
    "Dynamic_Text_Conditioning": Dynamic_Text_Conditioning,
    "Text_Image_Conditioning_Combine": Text_Image_Conditioning_Combine,
    "Dynamic_Sampler": Dynamic_Sampler,
    "sample_decode": sample_decode
    }


NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamiCrafterModelLoader": "DynamiCrafterModelLoader_qsn",
    "Dynamic_dtype": "Dynamic_dtype_qsn",
    "Dynamic_encode": "Dynamic_encode_qsn",
    "Dynamic_Image_Conditioning": "Dynamic_Image_Conditioning_qsn",
    "Dynamic_Text_Conditioning": "Dynamic_Text_Conditioning_qsn",
    "Text_Image_Conditioning_Combine": "Text_Image_Conditioning_Combine_qsn",
    "Dynamic_Sampler": "Dynamic_Sampler_qsn",
    "sample_decode": "sample_decode_qsn"
    }

