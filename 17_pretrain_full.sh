for i in 66
do
CUDA_VISIBLE_DEVICES=1 python MAESC_training.py \
      --dataset twitter17 ./src/data/jsons/twitter17_info.json \
      --checkpoint_dir ./ \
      --model_config config/pretrain_base.json \
      --lambda_init 0.45 \
      --curriculum_pace square \
      --image_text_similarity_path ./src/data/jsons/amended_similarity_by_whole2017.json \
      --image_text_region_similarity_path ./src/data/jsons/amended_similarity_by_region2017.json \
      --log_dir 17_aesc \
      --num_beams 4 \
      --eval_every 1 \
      --lr 4e-5 \
      --batch_size 16  \
      --epochs 55 \
      --grad_clip 5 \
      --warmup 0.1 \
      --seed $i \
      --checkpoint ./MABSA_VLP/checkpoint/pytorch_model.bin
done
