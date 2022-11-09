"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util
import numpy as np
from PIL import Image
import nibabel as nib

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    print('opt:', opt)

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    print('length dataset;', len(dataset))

    recon_imgs = []
    for i, data in enumerate(dataset):
        '''
        data key: dict_keys(['A', 'B', 'A_paths', 'B_paths'])
        # visual: dict_keys(['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B'])
        '''

        # print('data:', i ,'-', data)
        # print('data type:', type(data))
        # print('data key:', data.keys())

        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
            # breakr
        model.set_input(data)  # unpack data from data loader
        # print('test.py> set_input(data):', data)
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        
        '''
        print('test_py > visuals[rec_B]:', visuals['rec_B'])
        print(np.array(visuals['rec_B'].cpu()))
        print(len(np.array(visuals['rec_B'].cpu())))
        print('shape:', np.array(visuals['rec_B'].cpu()).shape)
        print('reshape:', (np.array(visuals['rec_B'].cpu()).reshape(256,256)).shape)
        recon_img = np.array(visuals['rec_B'].cpu()).reshape(256,256)
        recon_img = np.array(visuals['rec_B'].cpu()).reshape(256,256)
        '''
        # print(visuals.keys())
        # image_rec_med = util.tensor2im_med(visuals['rec_B'], imtype=np.uint8)
        # image_real_med = util.tensor2im_med(visuals['real_B'], imtype=np.uint8)
        # print('='*10, ' tensor2im_med ', '='*10)
        # print('image_rec_med:', image_rec_med.shape, np.min(image_rec_med), np.max(image_rec_med), np.mean(image_rec_med), np.std(image_rec_med))
        # print('image_real_med:', image_real_med.shape, np.min(image_real_med), np.max(image_real_med), np.mean(image_real_med), np.std(image_real_med))
        # print(visuals)

        image_rec = util.tensor2im(visuals['rec_B'], imtype=np.uint8)
        image_real = util.tensor2im(visuals['real_B'], imtype=np.uint8)
        # print('='*10, ' tensor2im ', '='*10)
        # print('image_rec:', image_rec.shape, np.min(image_rec), np.max(image_rec), np.mean(image_rec), np.std(image_rec))
        # print('image_real:', image_real.shape, np.min(image_real), np.max(image_real), np.mean(image_real), np.std(image_real))



        # print(image_real.shape)
        # image_real = Image.fromarray(image_real)
        # break
        pet_path = data['B_paths']
        img_path = model.get_image_paths()     # get image paths 
        
        # B_paths
        # break
        
        # print(len(img_path)) # 40448
        # break
        
        # tmp_img_real_path = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'predtmp0914').replace('MRI','realPET') +'.npy'
        # tmp_img_rec_path = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'predtmp0914').replace('MRI','recPET') +'.npy'
        
        #print(tmp_img_real_path)

        
        # tmp_img_real_path_med = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'predtmp0914_med').replace('MRI','realPET_med') +'.npy'
        # tmp_img_rec_path_med = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'predtmp0914_med').replace('MRI','recPET_med') +'.npy'

        # np.save(tmp_img_real_path_med, image_real_med)
        # np.save(tmp_img_rec_path_med, image_rec_med)


        # tmp_img_real_path = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'predtmp0914_raw').replace('MRI','realPET') +'.npy'
        # tmp_img_rec_path = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'predtmp0914_raw').replace('MRI','recPET') +'.npy'

        # np.save(tmp_img_real_path, image_real)
        # np.save(tmp_img_rec_path, image_rec)
        # print(image_rec.shape)
        # util.save_image(image_real, tmp_img_real_path, aspect_ratio=1.0)
        # util.save_image(image_rec, tmp_img_rec_path, aspect_ratio=1.0)

        # image_real.save(f"{tmp_img_path}.png")

        # util.save_image(image_numpy, image_path, aspect_ratio=1.0)

        # util.save_image(im)
        if int(''.join(img_path[i]).split('_')[-1].split('.nii')[0]) == 0:
            npy_nii_rec = util.pre_save_image(image_rec, aspect_ratio=1.0)
            npy_nii_real = util.pre_save_image(image_real, aspect_ratio=1.0)

        elif int(''.join(img_path[i]).split('_')[-1].split('.nii')[0]) < 255:
            tmp_rec = util.pre_save_image(image_rec, aspect_ratio=1.0)            
            tmp_real = util.pre_save_image(image_real, aspect_ratio=1.0)
            
            npy_nii_rec = np.concatenate((npy_nii_rec, tmp_rec), axis=1)
            npy_nii_real = np.concatenate((npy_nii_real, tmp_real), axis=1)   
        
        else: #int(''.join(img_path[i]).split('_')[-1].split('.nii')[0]) == 255:
            tmp_rec = util.pre_save_image(image_rec, aspect_ratio=1.0)            
            tmp_real = util.pre_save_image(image_real, aspect_ratio=1.0)

            npy_nii_rec = np.concatenate((npy_nii_rec, tmp_rec), axis=1)
            npy_nii_real = np.concatenate((npy_nii_real, tmp_real), axis=1)      
        
            nii_img_rec_path = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'pred_rec_1020_70').replace('MRI','recPET').replace('_255','') +'.nii'
            nii_img_real_path = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'pred_rec_1020_70').replace('MRI','realPET').replace('_255','') +'.nii'

            # image_path.split()
            img_affine_path = ''.join(pet_path[i]).split('_255.nii')[0]+'.nii'
            img_affine = nib.load(img_affine_path).affine
            # print(img_affine)

            rec_nii_img = nib.Nifti1Image(npy_nii_rec, affine=img_affine)  
            real_nii_img = nib.Nifti1Image(npy_nii_real, affine=img_affine)

            # rec_nii_img = nib.Nifti1Image(npy_nii_rec, affine=np.eye(4))
            # real_nii_img = nib.Nifti1Image(npy_nii_real, affine=np.eye(4))

            nib.save(rec_nii_img, nii_img_rec_path)           
            nib.save(real_nii_img, nii_img_real_path)           

            # break
        
        # util.tensor2im(visuals['rec_B'], imtype=np.uint8)
        
        # print('test_py > visuals[1]:', visuals[1])
 
        # img_path = model.get_image_paths()     # get image paths
    
        # tmp_img_path = ''.join(img_path[i]).split('.nii')[0].replace('testA', 'predB') +'.png'
        '''    
        # print('test_py > img_path:', img_path)
        print('test_py > img_path.TYPE :', type(img_path))
        print('test_py > img_path.len :', len(img_path))
        print('test_py > img_path :', img_path[i])
        print('img num:', ''.join(img_path[i]).split('_')[-1].split('.nii')[0], )
        print(''.join(img_path[i]).split('.nii')[0].replace('testA', 'predB') +'.npy')

        recon_imgs.append(recon_img)
        '''

        # util.save_image(image_numpy, tmp_img_path, aspect_ratio=1.0)
        # if int(''.join(img_path[i]).split('_')[-1].split('.nii')[0]) == 255:
        # print('test_py > img_path.shape :', img_path.shape)
            # np.save(''.join(img_path[i]).split('.nii')[0].replace('testA', 'predB') +'.npy', recon_imgs)
            # print(i)
            # break
            # recon_imgs = []
        # if i % 5 == 0:  # save images to an HTML file
                # print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    # webpage.save()  # save the HTML
