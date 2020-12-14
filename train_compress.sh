# python train_compress.py --dataroot dataset/dh/vox_rot --batch_size=9 --name compress_sdf --log_path log/compress_sdf --gpu_ids 1,2,3 --num_threads=5  \
# --load_netG_checkpoint_path /media/data1/shaoruizhi/3DGAN/checkpoints/compress_sdf/netG_latest

python train_compress.py --dataroot dataset/dh/vox --batch_size=1 --name compress_sdf --log_path log/compress_sdf --gpu_ids 0 --num_threads=5  \
--load_netG_checkpoint_path /media/data1/shaoruizhi/3DGAN/checkpoints/compress_sdf/netG_latest --show

