local model_name = 'runwayml/stable-diffusion-v1-5';

local seed = 19950815;
local instance_prompt = '<cat-toy> toy';
local is_prior_preservation = true;
local prior_preservation_class_prompt = 'a photo of a cat clay toy';
local prior_loss_weight = 0.5;
local class_images_dir = './class-images';
local num_class_images = 12;
local sample_batch_size = 2;
local resolution = 512;

local is_train_text_encoder = false;
local is_gradient_checkpointing = true;
local learning_rate = 5e-06;
local train_steps = 300;
local train_batch_size = 2;
local grad_accum = 2;
local grad_norm = 1.0;
local is_mixed_precision = true;
local save_steps = 50;
local devices = 1;

{
    steps: {
        generate_class_image: {
            type: 'generate_class_image',
            is_prior_preservation: is_prior_preservation,
            class_images_dir: class_images_dir,
            num_class_images: num_class_images,
            sample_batch_size: sample_batch_size,
            model_name: model_name,
            prior_preservation_class_prompt: prior_preservation_class_prompt,
        },
        raw_instance_data: {
            type: 'datasets::load',
            path: 'diffusers/cat_toy_example',
        },
        transform_data: {
            type: 'transform_data',
            instance_dataset: { type: 'ref', ref: 'raw_instance_data' },
            tokenizer: {
                pretrained_model_name_or_path: model_name,
                subfolder: 'tokenizer',
            },
            instance_prompt: instance_prompt,
            image_size: resolution,
            class_prompt: prior_preservation_class_prompt,
            class_dataset: { type: 'ref', ref: 'generate_class_image' },
        },
        train_model: {
            type: 'torch::train',
            seed: seed,
            training_engine: {
                amp: is_mixed_precision,
                max_grad_norm: grad_norm,
                optimizer: {
                    type: 'torch::AdamW',
                    lr: 5e-06,
                },
                lr_scheduler: {
                    type: 'diffusers::constant',
                },
            },
            model: {
                type: 'stable_diffusion',
                model_name: model_name,
                is_prior_preservation: is_prior_preservation,
                is_train_text_encoder: is_train_text_encoder,
                is_gradient_checkpointing: is_gradient_checkpointing,
                prior_loss_weight: prior_loss_weight,
            },
            dataset_dict: { type: 'ref', ref: 'transform_data' },
            train_dataloader: {
                shuffle: true,
                batch_size: train_batch_size,
                collate_fn: {
                    type: 'custom_collator',
                    tokenizer: {
                        pretrained_model_name_or_path: model_name,
                        subfolder: 'tokenizer',
                    },
                    is_prior_preservation: is_prior_preservation,
                },
            },
            train_steps: train_steps,
            grad_accum: grad_accum,
            device_count: devices,
            checkpoint_every: save_steps,
        },
        create_pipeline: {
            type: 'create_pipeline',
            model_name: model_name,
            model: { type: 'ref', ref: 'train_model' },
        },
        generate_new_image: {
            type: 'generate_new_images',
            pipe: { type: 'ref', ref: 'create_pipeline' },
            prompt: 'a <cat-toy> in mad max fury road',
            seed: seed,
            generated_image_path: 'cat-mad-max.png',
        },
    },
}
