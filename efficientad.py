#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
import itertools
import os
import random
import json
from datetime import datetime
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output')
    parser.add_argument('-m', '--model_size', default='medium',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default=None,
                        help='Path to teacher weights. Auto-selected based on model_size if not provided')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    # Data augmentation parameters
    parser.add_argument('--augment_multiplier', type=int, default=4,
                        help='Number of augmented images generated per original image')
    parser.add_argument('--rotation_range', type=float, default=15.0,
                        help='Maximum rotation angle in degrees for augmentation')
    parser.add_argument('--enable_rotation', action='store_true', default=True,
                        help='Enable rotation augmentation')
    parser.add_argument('--brightness', type=float, default=0.3,
                        help='Brightness jitter factor (0-1)')
    parser.add_argument('--contrast', type=float, default=0.3,
                        help='Contrast jitter factor (0-1)')
    parser.add_argument('--saturation', type=float, default=0.3,
                        help='Saturation jitter factor (0-1)')
    parser.add_argument('--hue', type=float, default=0.1,
                        help='Hue jitter factor (0-0.5)')
    parser.add_argument('--enable_blur', action='store_true', default=True,
                        help='Enable Gaussian blur augmentation')
    parser.add_argument('--enable_flip', action='store_true', default=True,
                        help='Enable horizontal/vertical flip augmentation')
    parser.add_argument('--enable_perspective', action='store_true', default=True,
                        help='Enable perspective transform augmentation')
    parser.add_argument('--run_name', default=None,
                        help='Custom name for this run. Auto-generated if not provided')
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_augmentation_transform(rotation_range=15.0, enable_rotation=True, 
                                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1,
                                enable_blur=True, enable_flip=True, enable_perspective=True):
    """Create comprehensive augmentation transform with multiple options.
    
    Args:
        rotation_range: Maximum rotation angle in degrees
        enable_rotation: Enable rotation augmentation
        brightness: Brightness jitter factor (0-1)
        contrast: Contrast jitter factor (0-1)
        saturation: Saturation jitter factor (0-1)
        hue: Hue jitter factor (0-0.5)
        enable_blur: Enable Gaussian blur
        enable_flip: Enable horizontal/vertical flips
        enable_perspective: Enable perspective transforms
    """
    aug_transforms = []
    
    # Color augmentations - individual
    aug_transforms.extend([
        transforms.ColorJitter(brightness=brightness),
        transforms.ColorJitter(contrast=contrast),
        transforms.ColorJitter(saturation=saturation),
    ])
    
    # Combined color jitter
    aug_transforms.append(
        transforms.ColorJitter(
            brightness=brightness * 0.7,
            contrast=contrast * 0.7,
            saturation=saturation * 0.7,
            hue=hue
        )
    )
    
    # Grayscale conversion (partial)
    aug_transforms.append(transforms.RandomGrayscale(p=1.0))
    
    # Gaussian blur
    if enable_blur:
        aug_transforms.extend([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
        ])
    
    # Rotation augmentations
    if enable_rotation:
        aug_transforms.extend([
            transforms.RandomRotation(degrees=rotation_range),
            transforms.RandomRotation(degrees=rotation_range * 0.5),  # Subtle rotation
            # Rotation + color
            transforms.Compose([
                transforms.RandomRotation(degrees=rotation_range),
                transforms.ColorJitter(brightness=brightness * 0.5, contrast=contrast * 0.5)
            ]),
        ])
    
    # Flip augmentations
    if enable_flip:
        aug_transforms.extend([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            # Flip + color
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=brightness * 0.5, saturation=saturation * 0.5)
            ]),
        ])
    
    # Perspective and affine transforms
    if enable_perspective:
        aug_transforms.extend([
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
            transforms.RandomAffine(
                degrees=rotation_range * 0.5 if enable_rotation else 0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5
            ),
        ])
    
    # Combined augmentations for more variety
    aug_transforms.extend([
        # Color + blur
        transforms.Compose([
            transforms.ColorJitter(brightness=brightness * 0.5, contrast=contrast * 0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)) if enable_blur else transforms.Lambda(lambda x: x)
        ]),
        # Rotation + perspective
        transforms.Compose([
            transforms.RandomRotation(degrees=rotation_range * 0.5) if enable_rotation else transforms.Lambda(lambda x: x),
            transforms.RandomPerspective(distortion_scale=0.15, p=1.0) if enable_perspective else transforms.Lambda(lambda x: x)
        ]),
        # Sharpness adjustment
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0),
        transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=1.0),
        # Posterize (reduce color depth)
        transforms.RandomPosterize(bits=6, p=1.0),
        # Autocontrast
        transforms.RandomAutocontrast(p=1.0),
        # Equalize histogram
        transforms.RandomEqualize(p=1.0),
    ])
    
    return transforms.RandomChoice(aug_transforms)

# Default transform_ae (will be updated in main with config params)
transform_ae = get_augmentation_transform()

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))


class AugmentedDataset(Dataset):
    """Dataset wrapper that multiplies images with augmentations."""
    def __init__(self, base_dataset, augment_multiplier=1, rotation_range=15.0, enable_rotation=True,
                 brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1,
                 enable_blur=True, enable_flip=True, enable_perspective=True):
        self.base_dataset = base_dataset
        self.augment_multiplier = augment_multiplier
        self.augment_transform = get_augmentation_transform(
            rotation_range=rotation_range, enable_rotation=enable_rotation,
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
            enable_blur=enable_blur, enable_flip=enable_flip, enable_perspective=enable_perspective
        )
        
    def __len__(self):
        return len(self.base_dataset) * self.augment_multiplier
    
    def __getitem__(self, index):
        base_idx = index // self.augment_multiplier
        aug_idx = index % self.augment_multiplier
        
        # Get original item
        item = self.base_dataset[base_idx]
        
        if aug_idx == 0:
            # Return original (no extra augmentation beyond base transform)
            return item
        else:
            # Apply additional rotation/augmentation
            # The base_dataset already applies train_transform
            return item  # Augmentation is built into the transform

def get_run_id(run_name=None):
    """Generate a unique run identifier."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if run_name:
        return f"{run_name}_{timestamp}"
    return timestamp


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()
    
    # Auto-select weights based on model size if not provided
    if config.weights is None:
        if config.model_size == 'small':
            config.weights = 'models/teacher_small.pth'
        elif config.model_size == 'medium':
            config.weights = 'models/teacher_medium.pth'
        print(f"Auto-selected weights: {config.weights}")
    
    # Update global transform_ae with config parameters
    global transform_ae
    transform_ae = get_augmentation_transform(
        rotation_range=config.rotation_range,
        enable_rotation=config.enable_rotation,
        brightness=config.brightness,
        contrast=config.contrast,
        saturation=config.saturation,
        hue=config.hue,
        enable_blur=config.enable_blur,
        enable_flip=config.enable_flip,
        enable_perspective=config.enable_perspective
    )

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # Generate unique run ID
    run_id = get_run_id(config.run_name)
    
    # create output dir with unique run folder
    run_output_dir = os.path.join(config.output_dir, 'runs', run_id)
    train_output_dir = os.path.join(run_output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(run_output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Save run configuration
    config_dict = vars(config).copy()
    config_dict['run_id'] = run_id
    config_dict['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(run_output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Model Size: {config.model_size}")
    print(f"Dataset: {config.dataset}/{config.subdataset}")
    print(f"Augmentation Multiplier: {config.augment_multiplier}x")
    print(f"Augmentation Settings:")
    print(f"  - Rotation: {'ON' if config.enable_rotation else 'OFF'} (±{config.rotation_range}°)")
    print(f"  - Brightness: {config.brightness}, Contrast: {config.contrast}")
    print(f"  - Saturation: {config.saturation}, Hue: {config.hue}")
    print(f"  - Blur: {'ON' if config.enable_blur else 'OFF'}")
    print(f"  - Flip: {'ON' if config.enable_flip else 'OFF'}")
    print(f"  - Perspective: {'ON' if config.enable_perspective else 'OFF'}")
    print(f"Output Directory: {run_output_dir}")
    print(f"{'='*60}\n")

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')
    
    # Apply augmentation multiplier
    if config.augment_multiplier > 1:
        train_set = AugmentedDataset(
            train_set, 
            augment_multiplier=config.augment_multiplier,
            rotation_range=config.rotation_range,
            enable_rotation=config.enable_rotation,
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue,
            enable_blur=config.enable_blur,
            enable_flip=config.enable_flip,
            enable_perspective=config.enable_perspective
        )
        print(f"Training set size: {len(train_set)} (with {config.augment_multiplier}x augmentation)")
    else:
        print(f"Training set size: {len(train_set)}")


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
    tqdm_obj = tqdm(range(config.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output)**2
        distance_stae = (ae_output - student_output_ae)**2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                "Current loss: {:.4f}  ".format(loss_total.item()))

        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir,
                                             'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                             'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir,
                                                 'autoencoder_tmp.pth'))

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')
            auc = test(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=None, desc='Intermediate inference')
            print('Intermediate image auc: {:.4f}'.format(auc))

            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                         'autoencoder_final.pth'))
    
    # Save normalization parameters for inference
    norm_params = {
        'teacher_mean': teacher_mean.cpu().numpy().tolist(),
        'teacher_std': teacher_std.cpu().numpy().tolist()
    }
    with open(os.path.join(train_output_dir, 'normalization_params.json'), 'w') as f:
        json.dump(norm_params, f)

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    
    # Save quantile parameters
    quantile_params = {
        'q_st_start': float(q_st_start.cpu().numpy()),
        'q_st_end': float(q_st_end.cpu().numpy()),
        'q_ae_start': float(q_ae_start.cpu().numpy()),
        'q_ae_end': float(q_ae_end.cpu().numpy())
    }
    with open(os.path.join(train_output_dir, 'quantile_params.json'), 'w') as f:
        json.dump(quantile_params, f)
    
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    
    print(f'\nTraining completed! Run ID: {run_id}')
    print(f'Models and results saved to: {run_output_dir}')

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    """Run inference on test set and compute comprehensive metrics."""
    y_true = []
    y_score = []
    y_score_mean = []  # Mean anomaly score
    y_score_percentile = []  # 99th percentile anomaly score
    defect_scores = {}  # Scores per defect class
    
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_score_mean_image = np.mean(map_combined)
        y_score_percentile_image = np.percentile(map_combined, 99)
        
        y_true.append(y_true_image)
        y_score.append(y_score_image)
        y_score_mean.append(y_score_mean_image)
        y_score_percentile.append(y_score_percentile_image)
        
        # Track scores per defect class
        if defect_class not in defect_scores:
            defect_scores[defect_class] = []
        defect_scores[defect_class].append(y_score_image)
    
    # Compute comprehensive metrics
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    auc_mean = roc_auc_score(y_true=y_true, y_score=y_score_mean)
    auc_percentile = roc_auc_score(y_true=y_true, y_score=y_score_percentile)
    
    # Average Precision (AP)
    ap = average_precision_score(y_true=y_true, y_score=y_score)
    
    # Precision-Recall at optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]
    
    # Create results dictionary
    results = {
        'image_auc': auc * 100,
        'image_auc_mean': auc_mean * 100,
        'image_auc_percentile': auc_percentile * 100,
        'average_precision': ap * 100,
        'best_f1': best_f1 * 100,
        'best_threshold': float(best_threshold),
        'precision_at_best_f1': best_precision * 100,
        'recall_at_best_f1': best_recall * 100,
        'defect_class_stats': {}
    }
    
    # Per-class statistics
    for defect_class, scores in defect_scores.items():
        results['defect_class_stats'][defect_class] = {
            'count': len(scores),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores))
        }
    
    # Save detailed results if output dir provided
    if test_output_dir is not None:
        results_file = os.path.join(os.path.dirname(test_output_dir), 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("Test Results Summary:")
    print(f"{'='*50}")
    print(f"  Image AUC (max):        {results['image_auc']:.2f}%")
    print(f"  Image AUC (mean):       {results['image_auc_mean']:.2f}%")
    print(f"  Image AUC (99th pct):   {results['image_auc_percentile']:.2f}%")
    print(f"  Average Precision:      {results['average_precision']:.2f}%")
    print(f"  Best F1 Score:          {results['best_f1']:.2f}%")
    print(f"  Optimal Threshold:      {results['best_threshold']:.4f}")
    print(f"  Precision @ Best F1:    {results['precision_at_best_f1']:.2f}%")
    print(f"  Recall @ Best F1:       {results['recall_at_best_f1']:.2f}%")
    print(f"\nPer-class Statistics:")
    for defect_class, stats in results['defect_class_stats'].items():
        status = 'GOOD' if defect_class == 'good' else 'DEFECT'
        print(f"  {defect_class} ({status}): n={stats['count']}, "
              f"mean={stats['mean_score']:.4f}, std={stats['std_score']:.4f}")
    print(f"{'='*50}\n")
    
    return results['image_auc']

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
