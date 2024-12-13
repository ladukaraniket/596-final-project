
import os
import sys
sys.path.append('./ai-toolkit')
from toolkit.job import run_job
from collections import OrderedDict
from PIL import Image
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from collections import OrderedDict

job_to_run = OrderedDict([
    ('job', 'extension'),
    ('config', OrderedDict([
        # this name will be the folder and filename name
        ('name', 'my_first_flux_lora_v1'),
        ('process', [
            OrderedDict([
                ('type', 'sd_trainer'),
                ('log_dir', './logs'),
                ('log_config', OrderedDict([
                    ('log_interval', 10)
                ])),
                # root folder to save training sessions/samples/weights
                ('training_folder', './output'),
                # uncomment to see performance stats in the terminal every N steps
                #('performance_log_every', 1000),
                ('device', 'cuda:0'),
                # if a trigger word is specified, it will be added to captions of training data if it does not already exist
                # alternatively, in your captions you can add [trigger] and it will be replaced with the trigger word
                # ('trigger_word', 'image'),
                ('network', OrderedDict([
                    ('type', 'lora'),
                    ('linear', 16),
                    ('linear_alpha', 16)
                ])),
                ('save', OrderedDict([
                    ('dtype', 'float16'),  # precision to save
                    ('save_every', 250),  # save every this many steps
                    ('max_step_saves_to_keep', 4)  # how many intermittent saves to keep
                ])),
                ('datasets', [
                    # datasets are a folder of images. captions need to be txt files with the same name as the image
                    # for instance image2.jpg and image2.txt. Only jpg, jpeg, and png are supported currently
                    # images will automatically be resized and bucketed into the resolution specified
                    OrderedDict([
                        ('folder_path', './dataset'),
                        ('caption_ext', 'txt'),
                        ('caption_dropout_rate', 0.05),  # will drop out the caption 5% of time
                        ('shuffle_tokens', False),  # shuffle caption order, split by commas
                        ('cache_latents_to_disk', True),  # leave this true unless you know what you're doing
                        ('resolution', [512, 768, 1024])  # flux enjoys multiple resolutions
                    ])
                ]),
                ('train', OrderedDict([
                    ('batch_size', 1),
                    ('steps', 500),  # total number of steps to train 500 - 4000 is a good range
                    ('gradient_accumulation_steps', 1),
                    ('train_unet', True),
                    ('train_text_encoder', False),  # probably won't work with flux
                    ('gradient_checkpointing', True),  # need the on unless you have a ton of vram
                    ('noise_scheduler', 'flowmatch'),  # for training only
                    ('optimizer', 'adamw8bit'),
                    ('lr', 1e-4),

                    # uncomment this to skip the pre training sample
                    # ('skip_first_sample', True),

                    # uncomment to completely disable sampling
                    # ('disable_sampling', True),

                    # uncomment to use new vell curved weighting. Experimental but may produce better results
                    # ('linear_timesteps', True),

                    # ema will smooth out learning, but could slow it down. Recommended to leave on.
                    ('ema_config', OrderedDict([
                        ('use_ema', True),
                        ('ema_decay', 0.99)
                    ])),

                    # will probably need this if gpu supports it for flux, other dtypes may not work correctly
                    ('dtype', 'bf16')
                ])),
                ('model', OrderedDict([
                    # huggingface model name or path
                    ('name_or_path', 'black-forest-labs/FLUX.1-schnell'),
                    ('assistant_lora_path', 'ostris/FLUX.1-schnell-training-adapter'), # Required for flux schnell training
                    ('is_flux', True),
                    ('quantize', True),  # run 8bit mixed precision
                    # low_vram is painfully slow to fuse in the adapter avoid it unless absolutely necessary
                    #('low_vram', True),  # uncomment this if the GPU is connected to your monitors. It will use less vram to quantize, but is slower.
                ])),
                ('sample', OrderedDict([
                    ('sampler', 'flowmatch'),  # must match train.noise_scheduler
                    ('sample_every', 250),  # sample every this many steps
                    ('width', 1024),
                    ('height', 1024),
                    ('prompts', [
                        # you can add [trigger] to the prompts here and it will be replaced with the trigger word
                        #'[trigger] holding a sign that says \'I LOVE PROMPTS!\'',
                        'woman with red hair, playing chess at the park, bomb going off in the background',
                        'a woman holding a coffee cup, in a beanie, sitting at a cafe',
                        'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
                        'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
                        'a bear building a log cabin in the snow covered mountains',
                        'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
                        'hipster man with a beard, building a chair, in a wood shop',
                        'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
                        'a man holding a sign that says, \'this is a sign\'',
                        'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle'
                    ]),
                    ('neg', ''),  # not used on flux
                    ('seed', 42),
                    ('walk_seed', True),
                    ('guidance_scale', 1), # schnell does not do guidance
                    ('sample_steps', 4) # 1 - 4 works well
                ]))
            ])
        ])
    ])),
    # you can add any additional meta info here. [name] is replaced with config name at top
    ('meta', OrderedDict([
        ('name', '[name]'),
        ('version', '1.0')
    ]))
])

run_job(job_to_run)