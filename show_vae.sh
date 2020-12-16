python show_vae.py --dataroot dataset/dh/vox --batch_size=20 --name vae_2d --log_path log/vae_2d --gpu_ids 2,3  --train_2d --num_threads=5 \
--load_netD_checkpoint_path /media/data1/shaoruizhi/3DGAN/checkpoints/vae_2d/netD_latest \
--load_netG_checkpoint_path /media/data1/shaoruizhi/3DGAN/checkpoints/vae_2d/netG_latest --show