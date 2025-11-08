from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import cv2
import os

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

def create_detailed_3d(image_path, output_name="3d_model"):
    """
    Convert single image to detailed 3D using Depth-Anything
    
    Args:
        image_path: Path to input image
        output_name: Name for output files (without extension)
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

def visualize_3d_surface(x, y, z, colors, title="3D Surface", ax=None, step=2, view_angle=(90, -90), z_limit=None, roll=0, flip_v=False, flip_h=False):
    """Create 3D surface visualization
    
    Args:
        x, y, z: Coordinate arrays
        colors: Color array
        title: Title for the plot
        ax: Matplotlib axis (if None, creates new figure)
        step: Subsampling step (lower = more detail)
        view_angle: (elevation, azimuth) for viewing angle
        z_limit: Optional tuple (min, max) to set z-axis limits
        roll: Roll angle in degrees (simulated by rotating data)
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
    
    # Remove the default flip - only apply flips when explicitly requested
    
    # Apply flips based on settings
    if flip_v:
        print("Applying vertical flip...")
        # Vertical flip (top-bottom)
        x_sub = np.flipud(x_sub)
        y_sub = np.flipud(y_sub)
        z_sub = np.flipud(z_sub)
        colors_sub = np.flipud(colors_sub)
    
    if flip_h:
        print("Applying horizontal flip (mirror)...")
        # Create mirror effect by inverting x coordinates
        x_max = x_sub.max()
        x_min = x_sub.min()
        x_sub = x_max + x_min - x_sub  # Invert x coordinates
        # Also flip the colors to match
        colors_sub = np.fliplr(colors_sub)
        z_sub = np.fliplr(z_sub)
    
    # Apply roll rotation if specified (rotate around z-axis)
    if roll != 0:
        # Convert roll to radians
        roll_rad = np.radians(roll)
        # Center the data
        x_center = (x_sub.max() + x_sub.min()) / 2
        y_center = (y_sub.max() + y_sub.min()) / 2
        # Translate to origin
        x_centered = x_sub - x_center
        y_centered = y_sub - y_center
        # Apply rotation matrix
        x_rotated = x_centered * np.cos(roll_rad) - y_centered * np.sin(roll_rad)
        y_rotated = x_centered * np.sin(roll_rad) + y_centered * np.cos(roll_rad)
        # Translate back
        x_sub = x_rotated + x_center
        y_sub = y_rotated + y_center
    
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
    print(f"Current view angle: elevation={view_angle[0]}°, azimuth={view_angle[1]}°, roll={roll}°")
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
    
    # Set z-axis limits if specified
    if z_limit is not None:
        ax.set_zlim(z_limit)
    
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

def save_mesh_ply(x, y, z, colors, filename, step=2):
    """Save PLY file with triangulated mesh and vertex colors"""
    print(f"Saving mesh PLY file: {filename}")
    
    # Create regular grid of vertices
    x_sub = x[::step, ::step]
    y_sub = y[::step, ::step] 
    z_sub = z[::step, ::step]
    colors_sub = colors[::step, ::step]
    
    # Flip vertically to match visualization
    x_sub = np.flipud(x_sub)
    y_sub = np.flipud(y_sub)
    z_sub = np.flipud(z_sub)
    colors_sub = np.flipud(colors_sub)
    
    rows, cols = x_sub.shape
    
    # Flatten vertices
    vertices = []
    vertex_colors = []
    
    for i in range(rows):
        for j in range(cols):
            vertices.append([x_sub[i,j], y_sub[i,j], z_sub[i,j]])
            vertex_colors.append(colors_sub[i,j])
    
    # Create triangular faces
    faces = []
    for i in range(rows-1):
        for j in range(cols-1):
            # Current vertex indices in flattened array
            v1 = i * cols + j
            v2 = i * cols + (j + 1)
            v3 = (i + 1) * cols + j
            v4 = (i + 1) * cols + (j + 1)
            
            # Create two triangles per quad
            faces.append([v1, v2, v3])  # First triangle
            faces.append([v2, v4, v3])  # Second triangle
    
    print(f"Saving {len(vertices)} vertices and {len(faces)} faces...")
    
    # Write PLY file
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        
        # Write vertices
        for i, vertex in enumerate(vertices):
            r, g, b = vertex_colors[i].astype(int)
            f.write(f'{vertex[0]} {vertex[1]} {vertex[2]} {r} {g} {b}\n')
        
        # Write faces
        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
    
    print(f"Mesh PLY file saved! Open in MeshLab, Blender, or CloudCompare")

def save_detailed_ply(x, y, z, colors, filename, step=2):
    """Save detailed PLY file for external 3D software (point cloud version)"""
    print(f"Saving point cloud PLY file: {filename.replace('.ply', '_points.ply')}")
    
    # Sample with specified step for detail control
    x_sub = x[::step, ::step]
    y_sub = y[::step, ::step]
    z_sub = z[::step, ::step]
    colors_sampled = colors[::step, ::step]
    
    # Flip vertically to match visualization
    x_sub = np.flipud(x_sub)
    y_sub = np.flipud(y_sub)
    z_sub = np.flipud(z_sub)
    colors_sampled = np.flipud(colors_sampled)
    
    # Flatten for point cloud
    x_flat = x_sub.flatten()
    y_flat = y_sub.flatten()
    z_flat = z_sub.flatten()
    
    # Check if colors array has correct shape
    if len(colors_sampled.shape) == 3 and colors_sampled.shape[-1] == 3:
        # RGB image
        colors_flat = colors_sampled.reshape(-1, 3)
    else:
        # Grayscale or unexpected format
        print(f"Color array shape for PLY: {colors_sampled.shape} - using grayscale")
        gray_values = colors_sampled.flatten()
        colors_flat = np.stack([gray_values]*3, axis=-1)
    
    print(f"Saving {len(x_flat)} points...")
    
    points_filename = filename.replace('.ply', '_points.ply')
    with open(points_filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(x_flat)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        for i in range(len(x_flat)):
            r, g, b = colors_flat[i].astype(int)
            f.write(f'{x_flat[i]} {y_flat[i]} {z_flat[i]} {r} {g} {b}\n')
    
    print(f"Point cloud PLY file saved!")

def save_depth_image(depth_map, filename):
    """Save depth map as image"""
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Depth image saved: {filename}")

def main():
    """Main function to process image and create 3D outputs"""
    
    # Configuration
    IMAGE_PATH = "reconstructed.png"  # Change this to your image path
    OUTPUT_NAME = "my_3d_model"
    
    VISUALIZATION_DETAIL = 1  # For 3D surface display (1=max detail, 2=half, 3=third, etc.)
    SAVE_DETAIL = 1           # For saved PLY files (1=max detail, 2=half, 3=third, etc.)

    DEFAULT_ELEVATION = 92
    DEFAULT_AZIMUTH = 90
    DEFAULT_ROLL = 0
    
    # Flip settings - set to True to flip the object
    FLIP_VERTICAL = False  # Set to True to flip vertically (top-bottom)
    FLIP_HORIZONTAL = True  # Set to True for mirror effect (left-right)

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file '{IMAGE_PATH}' not found!")
        return
    
    try:
        # Create detailed 3D data
        x, y, z, colors, original_depth = create_detailed_3d(IMAGE_PATH, OUTPUT_NAME)
        
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
                           view_angle=(DEFAULT_ELEVATION, DEFAULT_AZIMUTH),  # Use configured default view
                           roll=DEFAULT_ROLL,  # Apply roll rotation
                           flip_v=FLIP_VERTICAL,  # Apply vertical flip if enabled
                           flip_h=FLIP_HORIZONTAL)  # Apply horizontal flip if enabled
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()