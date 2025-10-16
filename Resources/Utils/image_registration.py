"""
Image Registration using SimpleITK
Core registration algorithms for 2D slice registration
"""
from __future__ import print_function
import SimpleITK as sitk
import numpy as np


class RegisterImages:
    """Handle 2D image registration using SimpleITK"""
    
    def __init__(self):
        self.fixed = None
        self.moving = None
        self.initial_transform = None
        self.deformable_transform = None
        self.verbose = False
        
    def display_images(self, fixed_npa, fixed, moving, registration_method, fn):
        """Display registration progress (for debugging)"""
        import matplotlib.pyplot as plt
        plt.subplots(1, 2, figsize=(10, 8))
        
        # Draw fixed image
        plt.subplot(1, 2, 1)
        plt.imshow(fixed_npa, cmap=plt.cm.Greys_r)
        plt.title('fixed image')
        plt.axis('off')
        
        print(registration_method.GetCurrentLevel(), 
              np.min(fixed_npa), np.max(fixed_npa))
        
        # Get current transform
        current_transform = sitk.Transform(registration_method.GetInitialTransform())
        current_transform.SetParameters(registration_method.GetOptimizerPosition())
        
        moving_transformed = sitk.Resample(moving, fixed, current_transform)
        moving_npa = sitk.GetArrayFromImage(moving_transformed)
        
        # Draw moving image difference
        plt.subplot(1, 2, 2)
        plt.imshow(moving_npa - fixed_npa, cmap=plt.cm.Greys_r)
        plt.title('moving image')
        plt.axis('off')
        
        fn_with_idx = f"{fn}_{len(self.metric_values):03d}.png"
        plt.savefig(fn_with_idx)
        plt.close()
    
    def start_plot(self):
        """Initialize plotting data"""
        self.metric_values = []
        self.multires_iterations = []
    
    def end_plot(self):
        """Clean up plotting data"""
        del self.metric_values
        del self.multires_iterations
        import matplotlib.pyplot as plt
        plt.close()
    
    def plot_values(self, registration_method, fn):
        """Plot metric values"""
        import matplotlib.pyplot as plt
        
        self.metric_values.append(registration_method.GetMetricValue())
        
        plt.plot(self.metric_values, 'r')
        plt.plot(self.multires_iterations, 
                [self.metric_values[index] for index in self.multires_iterations], 'b*')
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.savefig(fn)
    
    def get_values(self, registration_method):
        """Get current registration values"""
        print(f"RegisterImages: {len(self.metric_values)} "
              f"{registration_method.GetMetricValue()} "
              f"{registration_method.GetOptimizerPosition()} "
              f"{registration_method.GetOptimizerScales()}")
    
    def update_multires_iterations(self):
        """Update multi-resolution iteration markers"""
        self.multires_iterations.append(len(self.metric_values))
    
    def RegisterAffine(self, fixed_img, moving_img, initial_transf, 
                      n_iterations=250, idx=0, mode=0, mode_score=0, 
                      apply_tr=False, debug=False):
        """
        Register images using affine/rigid transformation
        
        Args:
            fixed_img: Fixed image (SimpleITK Image)
            moving_img: Moving image (SimpleITK Image)
            initial_transf: Initial transform
            n_iterations: Number of iterations
            idx: Index for debugging output
            mode: 0=affine, 1=rigid
            mode_score: 0=MSE, 1=mutual information
            apply_tr: Apply initial transform to moving image first
            debug: Enable debug output
            
        Returns:
            Final transform (SimpleITK Transform)
        """
        if debug:
            import time
            start_time = time.time()
            moving_resampled = sitk.Resample(moving_img, fixed_img,
                initial_transf, sitk.sitkLinear, 0.0, fixed_img.GetPixelID())
            sitk.WriteImage(moving_resampled, f'{idx}_moving.nii.gz')
            sitk.WriteImage(fixed_img, f'{idx}_fixed.nii.gz')
            
            print(f"mode_Tra: {mode}\nMode_score: {mode_score}\nApply Transform: {apply_tr}")
        
        # Build composite transform
        if not apply_tr:
            all_transf = []
            try:
                n = initial_transf.GetNumberOfTransforms()
                for i in range(n):
                    tr = initial_transf.GetNthTransform(i)
                    all_transf.append(tr)
            except Exception:
                all_transf.append(initial_transf)
            
            if mode == 1:  # Rigid
                all_transf.append(sitk.Euler2DTransform())
            else:  # Affine
                all_transf.append(sitk.AffineTransform(2))
            
            initial_transf = sitk.CompositeTransform(all_transf)
        else:
            moving_img = sitk.Resample(moving_img, fixed_img,
                initial_transf, sitk.sitkLinear, 0.0, fixed_img.GetPixelID())
            output_tr = initial_transf
            
            if mode == 1:  # Rigid
                initial_transf = sitk.Euler2DTransform()
            else:  # Affine
                initial_transf = sitk.AffineTransform(2)
        
        self.fixed = sitk.Cast(fixed_img, sitk.sitkFloat32)
        self.moving = sitk.Cast(moving_img, sitk.sitkFloat32)
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set metric
        if mode_score == 0:
            registration_method.SetMetricAsMeanSquares()
            if debug:
                print("Use MSE")
        else:
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
            if debug:
                print("Use mutual information")
        
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer settings
        convergence_window_size = 50
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.01,
            numberOfIterations=n_iterations,
            convergenceMinimumValue=1e-4,
            convergenceWindowSize=convergence_window_size
        )
        
        # Set optimizer scales
        if mode == 0:  # Affine + MSE
            registration_method.SetOptimizerScales([2000, 2000, 2000, 2000, 1, 1])
        elif mode == 1:  # Rigid + MSE
            registration_method.SetOptimizerScales([2000, 1, 1])
        else:  # Affine + MI
            registration_method.SetOptimizerScales([100, 100, 100, 100, 1, 1])
        
        # Multi-resolution settings
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[16, 8, 4])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        
        # Set initial transform
        registration_method.SetInitialTransform(initial_transf, inPlace=False)
        
        # Add debug callbacks
        if debug:
            self.metric_values = []
            self.multires_iterations = []
            registration_method.AddCommand(sitk.sitkStartEvent, self.start_plot)
            registration_method.AddCommand(sitk.sitkEndEvent, self.end_plot)
            registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, 
                                          self.update_multires_iterations)
            registration_method.AddCommand(sitk.sitkIterationEvent, 
                                          lambda: self.get_values(registration_method))
            registration_method.AddCommand(sitk.sitkIterationEvent,
                                          lambda: self.plot_values(registration_method, 
                                                                  f'{idx}_plot.png'))
        
        # Execute registration
        final_transform = registration_method.Execute(self.fixed, self.moving)
        
        if debug:
            print("Initial transform:", self.initial_transform)
            print("Optimized transform:", final_transform)
            print("Optimized transform (flattened):", final_transform.FlattenTransform())
            print(f"Final metric value: {registration_method.GetMetricValue()}")
            print(f"Optimizer's stopping condition: "
                  f"{registration_method.GetOptimizerStopConditionDescription()}")
            
            end_time = time.time()
            moving_resampled = sitk.Resample(self.moving, self.fixed,
                final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
            sitk.WriteImage(moving_resampled, f'{idx}_moved.nii.gz')
            print(f"Done Running Affine Registration in {(end_time-start_time)/60} (min)")
        
        if self.verbose:
            print("RegisterImages: Done Running Affine Registration!")
        
        # Return composite transform if needed
        if apply_tr:
            output_tr.AddTransform(final_transform)
            return output_tr
        else:
            return final_transform
        
    def RegisterAffineWithMasks(self, fixed_img, moving_img, fixed_mask, moving_mask,
                           initial_transf, n_iterations=250, idx=0, mode=0, 
                           mode_score=1, debug=False):
        """
        Register images using affine/rigid transformation with mask support
        
        Args:
            fixed_img: Fixed image (SimpleITK Image)
            moving_img: Moving image (SimpleITK Image)
            fixed_mask: Fixed mask (SimpleITK Image, sitkUInt8)
            moving_mask: Moving mask (SimpleITK Image, sitkUInt8, optional)
            initial_transf: Initial transform
            n_iterations: Number of iterations
            idx: Index for debugging output
            mode: 0=affine, 1=rigid
            mode_score: 0=MSE, 1=mutual information
            debug: Enable debug output
            
        Returns:
            Final transform (SimpleITK Transform)
        """
        if debug:
            import time
            start_time = time.time()
            moving_resampled = sitk.Resample(moving_img, fixed_img,
                initial_transf, sitk.sitkLinear, 0.0, fixed_img.GetPixelID())
            sitk.WriteImage(moving_resampled, f'{idx}_moving_masked.nii.gz')
            sitk.WriteImage(fixed_img, f'{idx}_fixed_masked.nii.gz')
            if fixed_mask is not None:
                sitk.WriteImage(fixed_mask, f'{idx}_fixed_mask.nii.gz')
            if moving_mask is not None:
                sitk.WriteImage(moving_mask, f'{idx}_moving_mask.nii.gz')
            
            print(f"mode_Tra: {mode}\nMode_score: {mode_score}")
        
        # Build composite transform
        all_transf = []
        try:
            n = initial_transf.GetNumberOfTransforms()
            for i in range(n):
                tr = initial_transf.GetNthTransform(i)
                all_transf.append(tr)
        except Exception:
            all_transf.append(initial_transf)
        
        if mode == 1:  # Rigid
            all_transf.append(sitk.Euler2DTransform())
        else:  # Affine
            all_transf.append(sitk.AffineTransform(2))
        
        initial_transf = sitk.CompositeTransform(all_transf)
        
        self.fixed = sitk.Cast(fixed_img, sitk.sitkFloat32)
        self.moving = sitk.Cast(moving_img, sitk.sitkFloat32)
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set metric - prefer Mutual Information with masks
        if mode_score == 0:
            registration_method.SetMetricAsMeanSquares()
            if debug:
                print("Use MSE")
        else:
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            if debug:
                print("Use mutual information")
        
        # Set masks if provided
        if fixed_mask is not None:
            registration_method.SetMetricFixedMask(fixed_mask)
            if self.verbose or debug:
                print("     - Fixed mask applied to registration")
        
        if moving_mask is not None:
            registration_method.SetMetricMovingMask(moving_mask)
            if self.verbose or debug:
                print("     - Moving mask applied to registration")
        
        # Increase sampling in masked regions
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.1)  # Increased from 0.05
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer settings - adjusted for mask-based registration
        convergence_window_size = 50
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.1,  # Slightly increased learning rate
            numberOfIterations=n_iterations,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=convergence_window_size,
            estimateLearningRate=registration_method.EachIteration  # Adaptive learning rate
        )
        
        # Set optimizer scales
        if mode == 0:  # Affine
            if mode_score == 0:  # MSE
                registration_method.SetOptimizerScales([2000, 2000, 2000, 2000, 1, 1])
            else:  # MI
                registration_method.SetOptimizerScales([100, 100, 100, 100, 1, 1])
        elif mode == 1:  # Rigid
            if mode_score == 0:  # MSE
                registration_method.SetOptimizerScales([2000, 1, 1])
            else:  # MI
                registration_method.SetOptimizerScales([100, 1, 1])
        
        # Multi-resolution settings - more levels for better convergence
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[16, 8, 4, 2])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        
        # Set initial transform
        registration_method.SetInitialTransform(initial_transf, inPlace=False)
        
        # Add debug callbacks
        if debug:
            self.metric_values = []
            self.multires_iterations = []
            registration_method.AddCommand(sitk.sitkStartEvent, self.start_plot)
            registration_method.AddCommand(sitk.sitkEndEvent, self.end_plot)
            registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, 
                                        self.update_multires_iterations)
            registration_method.AddCommand(sitk.sitkIterationEvent, 
                                        lambda: self.get_values(registration_method))
            registration_method.AddCommand(sitk.sitkIterationEvent,
                                        lambda: self.plot_values(registration_method, 
                                                                f'{idx}_plot_masked.png'))
        
        # Execute registration
        final_transform = registration_method.Execute(self.fixed, self.moving)
        
        if debug:
            print("Initial transform:", initial_transf)
            print("Optimized transform:", final_transform)
            print(f"Final metric value: {registration_method.GetMetricValue()}")
            print(f"Optimizer's stopping condition: "
                f"{registration_method.GetOptimizerStopConditionDescription()}")
            
            end_time = time.time()
            moving_resampled = sitk.Resample(self.moving, self.fixed,
                final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
            sitk.WriteImage(moving_resampled, f'{idx}_moved_masked.nii.gz')
            print(f"Done Running Affine Registration with Masks in {(end_time-start_time)/60} (min)")
        
        if self.verbose:
            print("RegisterImages: Done Running Affine Registration with Masks!")
        
        return final_transform
        
    def RegisterDeformable(self, fixed_img, moving_img, initial_transf,
                          dist_between_grid_points=10, n_iterations=10, 
                          idx=0, debug=False):
        """
        Register images using B-spline deformable registration
        
        Args:
            fixed_img: Fixed image (SimpleITK Image)
            moving_img: Moving image (SimpleITK Image)
            initial_transf: Initial transform
            dist_between_grid_points: Distance between B-spline grid points (mm)
            n_iterations: Number of iterations
            idx: Index for debugging output
            debug: Enable debug output
            
        Returns:
            Final B-spline transform (SimpleITK Transform)
        """
        self.fixed = sitk.Cast(fixed_img, sitk.sitkFloat32)
        self.moving = sitk.Cast(moving_img, sitk.sitkFloat32)
        
        # Apply initial transform to moving image
        moved = sitk.Resample(self.moving, self.fixed, initial_transf,
                             sitk.sitkLinear, 0.0, self.moving.GetPixelID())
        
        # Determine number of B-spline control points
        grid_physical_spacing = [dist_between_grid_points, dist_between_grid_points, 
                                dist_between_grid_points]
        image_physical_size = [size * spacing for size, spacing in 
                              zip(self.fixed.GetSize(), moved.GetSpacing())]
        mesh_size = [int(image_size / grid_spacing + 0.5)
                    for image_size, grid_spacing in 
                    zip(image_physical_size, grid_physical_spacing)]
        
        # Initialize B-spline transform
        initial_transform = sitk.BSplineTransformInitializer(
            image1=self.fixed,
            transformDomainMeshSize=mesh_size,
            order=3
        )
        
        if debug:
            sitk.WriteImage(moved, f'moving_deformable_{idx}.mha')
            sitk.WriteImage(self.fixed, f'fixed_deformable_{idx}.mha')
        
        # Setup registration method
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Optimizer settings
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=n_iterations
        )
        
        # Multi-resolution settings
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[16, 8, 4])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        
        registration_method.SetInitialTransform(initial_transform)
        
        # Execute registration
        final_transform = registration_method.Execute(self.fixed, moved)
        
        if debug:
            moving_resampled = sitk.Resample(moved, self.fixed,
                final_transform, sitk.sitkLinear, 0.0, self.moving.GetPixelID())
            print(np.max(sitk.GetArrayFromImage(moving_resampled)))
            sitk.WriteImage(moving_resampled, f'moved_deformable_{idx}.mha')
        
        if self.verbose:
            print("RegisterImages: Done Running Deformable Registration!")
        
        return final_transform
    
    