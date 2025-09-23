from __future__ import print_function
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import time

class RegisterImages:
    """
    A class for registering medical images using SimpleITK, with built-in
    plotting and debugging capabilities.
    """

    def __init__(self, verbose=False):
        self.fixed = None
        self.moving = None
        self.initial_transform = None
        self.deformable_transform = None
        self.verbose = verbose
        # Class-level variables to replace globals
        self.metric_values = []
        self.multires_iterations = []

    def _display_images(self, fixed_npa, fixed, moving, registration_method, fn):
        """Displays image differences during registration."""
        plt.subplots(1, 2, figsize=(10, 8))

        # Draw the fixed image in the first subplot.
        plt.subplot(1, 2, 1)
        plt.imshow(fixed_npa, cmap=plt.cm.Greys_r)
        plt.title('Fixed Image')
        plt.axis('off')

        if self.verbose:
            print(f"Current Level: {registration_method.GetCurrentLevel()}")

        current_transform = sitk.Transform(registration_method.GetInitialTransform())
        current_transform.SetParameters(registration_method.GetOptimizerPosition())

        moving_resampled = sitk.Resample(moving, fixed, current_transform)
        moving_npa = sitk.GetArrayFromImage(moving_resampled)
        
        # Draw the moving image in the second subplot.
        plt.subplot(1, 2, 2)
        plt.imshow(moving_npa - fixed_npa, cmap=plt.cm.Greys_r)
        plt.title('Moving Image (After Current Transform)')
        plt.axis('off')

        plot_filename = f"{fn}_{len(self.metric_values):03d}.png"
        plt.savefig(plot_filename)
        plt.close()

    def _display_images_with_alpha(self, image_z, alpha, fixed, moving):
        """Displays alpha blended images."""
        img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
        plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
        plt.axis('off')
        plt.show()

    def _start_plot(self):
        """Callback for registration StartEvent."""
        self.metric_values = []
        self.multires_iterations = []

    def _end_plot(self):
        """Callback for registration EndEvent."""
        self.metric_values = []
        self.multires_iterations = []
        plt.close()

    def _plot_values(self, registration_method, fn):
        """Callback for registration IterationEvent to plot metric values."""
        self.metric_values.append(registration_method.GetMetricValue())
        
        plt.plot(self.metric_values, 'r')
        plt.plot(self.multires_iterations,
                 [self.metric_values[index] for index in self.multires_iterations], 'b*')
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.savefig(fn)

    def _get_values(self, registration_method):
        """Callback for registration IterationEvent to print values."""
        if self.verbose:
            print(f"Iteration: {len(self.metric_values)}, Metric Value: {registration_method.GetMetricValue()}, "
                  f"Optimizer Position: {registration_method.GetOptimizerPosition()}")

    def _update_multires_iterations(self):
        """Callback for MultiResolutionIterationEvent."""
        self.multires_iterations.append(len(self.metric_values))

    def _attach_debug_commands(self, registration_method, idx):
        """Attaches all debugging callbacks to the registration method."""
        registration_method.AddCommand(sitk.sitkStartEvent, self._start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, self._end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, self._update_multires_iterations)
        registration_method.AddCommand(sitk.sitkIterationEvent,
                                     lambda: self._get_values(registration_method))
        registration_method.AddCommand(sitk.sitkIterationEvent,
                                     lambda: self._plot_values(registration_method, f'{idx}_plot.png'))
        # To avoid passing heavy images in the lambda, the original display_images call is commented out.
        # registration_method.AddCommand(sitk.sitkIterationEvent,
        #                                lambda: self._display_images(sitk.GetArrayFromImage(self.fixed),
        #                                                            self.fixed, self.moving, registration_method, f'{idx}_frame'))

    def register_affine(self, fixed_img, moving_img, initial_transf, nIterations=250, idx=0,
                        mode=0, mode_score=0, apply_tr=False, debug=False):
        """
        Performs affine registration between two SimpleITK images.

        Args:
            fixed_img (sitk.Image): The fixed image.
            moving_img (sitk.Image): The moving image.
            initial_transf (sitk.Transform): An initial transform to apply.
            nIterations (int): Number of optimizer iterations.
            idx (int): An index for output file naming.
            mode (int): 0 for Affine, 1 for Rigid.
            mode_score (int): 0 for Mean Squares, 1 for Mutual Information.
            apply_tr (bool): If True, composes a new transform with the initial one.
            debug (bool): Enables debugging and plotting callbacks.
        """
        start_time = time.time()

        if apply_tr:
            moved_img = sitk.Resample(moving_img, fixed_img, initial_transf,
                                      sitk.sitkLinear, 0.0, fixed_img.GetPixelID())
            output_tr = sitk.CompositeTransform(initial_transf)
            initial_transf = sitk.Euler2DTransform() if mode == 1 else sitk.AffineTransform(2)
            moving_img = moved_img
        else:
            all_transf = sitk.CompositeTransform([initial_transf]) if isinstance(initial_transf, sitk.CompositeTransform) else sitk.CompositeTransform([initial_transf])
            new_transform = sitk.Euler2DTransform() if mode == 1 else sitk.AffineTransform(2)
            all_transf.AddTransform(new_transform)
            initial_transf = all_transf

        self.fixed = sitk.Cast(fixed_img, sitk.sitkFloat32)
        self.moving = sitk.Cast(moving_img, sitk.sitkFloat32)

        registration_method = sitk.ImageRegistrationMethod()
        if mode_score == 0:
            registration_method.SetMetricAsMeanSquares()
        else:
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)

        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(learningRate=0.01,
                                                         numberOfIterations=nIterations,
                                                         convergenceMinimumValue=1e-4,
                                                         convergenceWindowSize=50)

        # Optimizer scales based on mode
        if mode == 0: # affine
            registration_method.SetOptimizerScales([2000, 2000, 2000, 2000, 1, 1])
        elif mode == 1: # rigid
            registration_method.SetOptimizerScales([2000, 1, 1])
        else: # affine with MI
            registration_method.SetOptimizerScales([100, 100, 100, 100, 1, 1])

        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[16, 8, 4])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        registration_method.SetInitialTransform(initial_transf, inPlace=False)

        if debug:
            self._attach_debug_commands(registration_method, idx)
        
        final_transform = registration_method.Execute(self.fixed, self.moving)

        if self.verbose:
            print(f"Optimizer's stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")
            print(f"Final metric value: {registration_method.GetMetricValue()}")
            print(f"Time taken: {(time.time() - start_time) / 60:.2f} min")
        
        return final_transform

    def register_deformable(self, fixed_img, moving_img, initial_transf,
                            dist_between_grid_points=10, nIterations=10, idx=0, debug=False):
        """
        Performs BSpline deformable registration.

        Args:
            fixed_img (sitk.Image): The fixed image.
            moving_img (sitk.Image): The moving image.
            initial_transf (sitk.Transform): An initial transform to apply before deformable registration.
            dist_between_grid_points (int): Physical spacing of the BSpline control grid in mm.
            nIterations (int): Number of optimizer iterations.
            idx (int): An index for output file naming.
            debug (bool): Enables debugging.
        """
        self.fixed = sitk.Cast(fixed_img, sitk.sitkFloat32)
        self.moving = sitk.Cast(moving_img, sitk.sitkFloat32)

        moved = sitk.Resample(self.moving, self.fixed, initial_transf,
                              sitk.sitkLinear, 0.0, self.moving.GetPixelID())

        grid_physical_spacing = [dist_between_grid_points, dist_between_grid_points]
        image_physical_size = [size * spacing for size, spacing in zip(self.fixed.GetSize(), moved.GetSpacing())]
        mesh_size = [int(image_size / grid_spacing + 0.5)
                     for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]

        bspline_transform = sitk.BSplineTransformInitializer(image1=self.fixed,
                                                             transformDomainMeshSize=mesh_size, order=3)

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.05)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=nIterations)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[16, 8, 4])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        registration_method.SetInitialTransform(bspline_transform)

        if debug:
            # The original code had some `sitk.WriteImage` calls for debugging.
            # We can put them here.
            sitk.WriteImage(moved, f'moving_deformable_{idx}.mha')
            sitk.WriteImage(self.fixed, f'fixed_deformable_{idx}.mha')

        final_transform = registration_method.Execute(self.fixed, moved)

        if debug:
            moved_resampled = sitk.Resample(moved, self.fixed, final_transform,
                                            sitk.sitkLinear, 0.0, self.moving.GetPixelID())
            sitk.WriteImage(moved_resampled, f'moved_deformable_{idx}.mha')

        if self.verbose:
            print("RegisterImages: Done Running Deformable Registration!")
            
        return final_transform
