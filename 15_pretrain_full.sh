CUDA_VISIBLE_DEVICES=3  python MAESC_training.py \
      --dataset twitter15 ./src/data/jsons/twitter15_info.json \
      --checkpoint_dir /home/data_ti6_d/lich/MABSA_VLP/checkpoint/ \
      --model_config config/pretrain_base.json \
      --lambda_init 0.7 \
      --curriculum_pace linear \
      --image_text_similarity_path /home/data_ti6_d/lich/CurriculumLearningMABSA/PreprocessData/amended_similarity_by_whole2015.json \
      --image_text_region_similarity_path /home/data_ti6_d/lich/CurriculumLearningMABSA/PreprocessData/amended_similarity_by_region2015.json \
      --log_dir 15_aesc \
      --num_beams 4 \
      --eval_every 1 \
      --lr 7e-5 \
      --batch_size 8  \
      --epochs 60 \
      --grad_clip 5 \
      --warmup 0.1 \
      --seed 66 \
      --checkpoint /home/data_ti6_d/lich/MABSA_VLP/checkpoint/pytorch_model.bin
