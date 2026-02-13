
import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a functional graph from the existing model layers
    # This avoids "sequential not called" errors by creating a fresh symbolic graph
    
    # 1. Create a new input tensor with the same shape as the image
    input_shape = img_array.shape[1:]
    img_input = tf.keras.Input(shape=input_shape)
    
    # 2. Pass this input through every layer in the model
    x = img_input
    last_conv_layer_output = None
    target_layer_found = False
    
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_layer_output = x
            target_layer_found = True
            
    if not target_layer_found:
         # Fallback search if names don't match exactly (though they should)
         raise ValueError(f"Layer {last_conv_layer_name} not found in model.")

    # 3. Create the Gradient model
    grad_model = tf.keras.models.Model(
        inputs=img_input,
        outputs=[last_conv_layer_output, x]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, output_path, confidence=None, label=None):
    # Load the original image
    img = cv2.imread(img_path)
    
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on original image
    superimposed_img = jet * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    # Threshold the heatmap to determine the bounding box
    # Use adaptive thresholding (50% of max intensity)
    max_val = np.max(heatmap)
    thresh_val = max_val * 0.5
    _, thresh = cv2.threshold(heatmap, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Find contours specific to the hot region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw on the superimposed image
    output_img = superimposed_img
    
    # Draw bounding box around the largest contour
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Draw thicker rectangle (Green for contrast on Jet heatmap)
        # Using pure green (0, 255, 0)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Add text
        if label and confidence:
            text = f"{label}: {confidence:.1%}" if isinstance(confidence, float) else f"{label}: {confidence}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw filled rectangle for text background
            # Position text above the box, or inside if near top edge
            text_x = x
            text_y = y - 10 if y - 10 > text_height else y + h + text_height + 10
            
            # Background rectangle coords
            bg_pt1 = (text_x, text_y - text_height - 5)
            bg_pt2 = (text_x + text_width, text_y + 5)
            
            # Draw black background
            cv2.rectangle(output_img, bg_pt1, bg_pt2, (0, 0, 0), -1)
            
            # Draw White Text
            cv2.putText(output_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Save the result
    cv2.imwrite(output_path, output_img)
