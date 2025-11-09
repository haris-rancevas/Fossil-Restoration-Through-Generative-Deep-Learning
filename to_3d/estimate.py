import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from transformers import pipeline
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter


def enhance_depth_detail(depth_map, remove_background=True):
    """Enhance depth map for better 3D detail"""
    # Convert to float32 for processing
    depth_enhanced = depth_map.astype(np.float32)
    
    # Apply bilateral filter to preserve edges while smoothing
    depth_filtered = cv2.bilateralFilter(depth_enhanced, 9, 75, 75)
    
    # Enhance contrast in depth
    depth_min, depth_max = depth_filtered.min(), depth_filtered.max()
    depth_contrast = (depth_filtered - depth_min) / (depth_max - depth_min)
    
    # Apply power curve to enhance depth differences
    depth_final = np.power(depth_contrast, 0.7) * 100
    
    if remove_background:
        # Invert depth so fossil protrudes upward (larger values = closer to viewer)
        depth_final = depth_final.max() - depth_final
        
        # Optional: enhance the depth range
        depth_final = depth_final * 1.5
    
    return depth_final

def create_detailed_3d(image_path):
    """
    Convert single image to detailed 3D using Depth-Anything
    
    Args:
        image_path: Path to input image
    """
    
    print("Loading Depth-Anything model...")
    # Use large model for better detail
    depth_estimator = pipeline('depth-estimation', 
                              model='depth-anything/Depth-Anything-V2-Large-hf')
    
    print(f"Processing image: {image_path}")
    # Load image at full resolution
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    original_size = image.size
    print(f"Original image size: {original_size}")
    
    # Get depth estimation
    result = depth_estimator(image)
    depth_map = np.array(result['depth'])
    
    print(f"Depth map size: {depth_map.shape}")
    
    # Resize image to match depth map
    img_array = np.array(image.resize((depth_map.shape[1], depth_map.shape[0])))
    
    # Ensure img_array is RGB
    if len(img_array.shape) == 2:
        # Grayscale - convert to RGB
        img_array = np.stack([img_array]*3, axis=-1)
    
    print(f"Image array shape: {img_array.shape}")
    
    # Enhance depth for better detail
    print("Enhancing depth detail...")
    z_enhanced = enhance_depth_detail(depth_map)
    
    # Create coordinate grids
    h, w = z_enhanced.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    return x, y, z_enhanced, img_array, depth_map

def visualize_3d_surface(x, y, z, colors, title="3D Surface", ax=None, step=2, view_angle=(90, -90)):
    """Create 3D surface visualization
    
    Args:
        x, y, z: Coordinate arrays
        colors: Color array
        title: Title for the plot
        ax: Matplotlib axis (if None, creates new figure)
        step: Subsampling step (lower = more detail)
        view_angle: (elevation, azimuth) for viewing angle
    """
    print(f"Creating 3D surface visualization (step={step} for detail)...")
    
    # Subsample for surface plot (adjust step for detail vs performance)
    x_sub = x[::step, ::step]
    y_sub = y[::step, ::step]
    z_sub = z[::step, ::step]
    colors_sub = colors[::step, ::step]
    
    # Create figure if ax not provided
    if ax is None:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Ensure colors are properly formatted
    if len(colors_sub.shape) == 2:
        # Grayscale - convert to RGB
        colors_rgb = np.stack([colors_sub, colors_sub, colors_sub], axis=-1)
    else:
        colors_rgb = colors_sub
    
    # Normalize colors to 0-1 range
    colors_normalized = colors_rgb.astype(np.float32) / 255.0
    
    # Swap axes to position X at top and Y at left when viewed from above
    # When looking from top (elev=90), X should run horizontally at top, Y vertically at left
    # We may need to swap or transpose based on the view
    
    # Create surface plot with colors - high resolution
    # Ensure fully opaque rendering
    surf = ax.plot_surface(x_sub, y_sub, z_sub, 
                          facecolors=colors_normalized,
                          alpha=1.0,  # Fully opaque
                          linewidth=0,  # No grid lines for cleaner look
                          antialiased=False,  # Disable antialiasing for sharper rendering
                          shade=False,  # Disable shading to preserve original colors
                          rstride=1,  # Row stride for higher resolution
                          cstride=1,  # Column stride for higher resolution
                          edgecolor='none',  # No edge color
                          rasterized=True)  # Rasterize for better opacity
    
    # Set axis labels with specific positions
    ax.set_xlabel('X (pixels)', labelpad=10)
    ax.set_ylabel('Y (pixels)', labelpad=10)
    ax.set_zlabel('Depth', labelpad=10)
    ax.set_title(title)
    
    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Note: 3D axes in matplotlib have limited label positioning options
    # The positions are controlled mainly by the viewing angle
    
    # Display rotation info
    print(f"Current view angle: elevation={view_angle[0]}°, azimuth={view_angle[1]}°")
    print("Rotate the 3D plot with mouse to find desired angle")
    print("Close the plot window to see final rotation values")
    
    # Better appearance
    ax.grid(True, alpha=0.3)  # Show grid with transparency
    
    # Adjust pane positions for better visibility
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make z-axis vertical and visible from top
    ax.zaxis.set_rotate_label(False)  # Don't rotate z label
    
    if ax is None:
        plt.tight_layout()
        plt.show()

def visualize_3d_points(x, y, z, colors, title="3D Point Cloud"):
    """Create 3D point cloud visualization"""
    print("Creating 3D point cloud visualization...")
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization (adjust step for detail vs performance)
    step = 4
    
    # Handle both grayscale and RGB images
    colors_sampled = colors[::step, ::step]
    
    # Check if colors array has correct shape
    if len(colors_sampled.shape) == 3 and colors_sampled.shape[-1] == 3:
        # RGB image
        colors_flat = colors_sampled.reshape(-1, 3)/255
    else:
        # Grayscale or unexpected format - use gray color
        print(f"Color array shape: {colors_sampled.shape} - using grayscale")
        gray_values = colors_sampled.flatten()/255
        colors_flat = np.stack([gray_values]*3, axis=-1)
    
    ax.scatter(x[::step, ::step].flatten(), 
              y[::step, ::step].flatten(), 
              z[::step, ::step].flatten(),
              c=colors_flat, 
              s=2,
              alpha=0.8)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Depth')
    ax.set_title(title)
    
    # Better viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to process image and create 3D outputs"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert 2D image to 3D depth visualization')
    parser.add_argument('--image', type=str, default='reconstructed.png',
                        help='Path to input image (default: reconstructed.png)')
    parser.add_argument('--detail', type=int, default=1,
                        help='Visualization detail level (1=max, 2=half, 3=third, etc.) (default: 1)')
    parser.add_argument('--elevation', type=float, default=90,
                        help='Default elevation angle for 3D view (default: 90)')
    parser.add_argument('--azimuth', type=float, default=90,
                        help='Default azimuth angle for 3D view (default: 90)')
    
    args = parser.parse_args()
    
    IMAGE_PATH = args.image
    VISUALIZATION_DETAIL = args.detail
    DEFAULT_ELEVATION = args.elevation
    DEFAULT_AZIMUTH = args.azimuth
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file '{IMAGE_PATH}' not found!")
        return
    
    try:
        # Create detailed 3D data
        x, y, z, colors, original_depth = create_detailed_3d(IMAGE_PATH)
        
        # First figure: Original image and depth map
        fig1 = plt.figure(figsize=(14, 6))
        
        # Original image
        ax1 = fig1.add_subplot(121)
        original_img = Image.open(IMAGE_PATH)
        ax1.imshow(original_img)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Depth map
        ax2 = fig1.add_subplot(122)
        im = ax2.imshow(original_depth, cmap='plasma')
        ax2.set_title('Depth Map', fontsize=14)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, label='Depth', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        # Second figure: 3D Surface as separate image
        fig2 = plt.figure(figsize=(10, 8))
        ax3 = fig2.add_subplot(111, projection='3d')
        
        # Add event handler to capture rotation on close
        def on_close(event):
            elev = ax3.elev
            azim = ax3.azim
            print(f"\nFinal rotation values when window closed:")
            print(f"  Elevation: {elev}°")
            print(f"  Azimuth: {azim}°")
            print(f"\nTo use this view as default, update lines 338-339 in the script:")
            print(f"  DEFAULT_ELEVATION = {elev}")
            print(f"  DEFAULT_AZIMUTH = {azim}")
        
        fig2.canvas.mpl_connect('close_event', on_close)
        
        visualize_3d_surface(x, y, z, colors, 
                           "3D Surface", 
                           ax=ax3, 
                           step=VISUALIZATION_DETAIL,  # Use configured detail level
                           view_angle=(DEFAULT_ELEVATION, DEFAULT_AZIMUTH)  # Use configured default view
        )
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()