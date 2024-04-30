import gradio as gr
import cv2
import numpy as np

def predict_single_action(video):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video: The video file uploaded by the user.
    '''

    CLASSES_LIST = ["Voilence", "NonViolence"] 

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video.name)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/20),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(20):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read() 

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (64, 64))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the pre-processed frames to the model and get the predicted probabilities.
    LRCN_model = load_model('C:/Users/aymer/Downloads/LRCN_model___Date_Time_2024_04_14__11_19_30___Loss_0.5454413294792175___Accuracy_0.7666666507720947.h5')
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
    # Release the VideoCapture object. 
    #video_reader.release()

    return f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}'

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_single_action, # The function to call
    inputs=gr.Video(sources="upload"), # Accept video files
    outputs= "text", # The output will be text (the prediction)
    title="Violent Video Prediction",
    description="Upload a video to get a prediction from the model."
)

# Launch the interface
iface.launch()
