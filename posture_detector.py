"""
YOLO-based Object and Posture Detection System
Detects people/objects and classifies basic postures (standing/sitting/bending)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


class PostureDetector:
    """
    Detects objects and classifies human postures using YOLO
    """
    
    def __init__(self, model_name='yolov8n-pose.pt'):
        """
        Initialize the detector with a YOLO model
        
        Args:
            model_name: YOLO model to use (yolov8n-pose.pt for pose detection)
        """
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        print("Model loaded successfully!")
        
    def classify_posture(self, keypoints):
        """
        Classify posture based on keypoint positions
        
        Args:
            keypoints: Array of keypoint coordinates [x, y, confidence]
            
        Returns:
            str: Posture classification (Standing/Sitting/Bending/Unknown)
        """
        if keypoints is None or len(keypoints) == 0:
            return "Unknown"
        
        # COCO keypoint indices
        # 5: left shoulder, 6: right shoulder
        # 11: left hip, 12: right hip
        # 13: left knee, 14: right knee
        # 15: left ankle, 16: right ankle
        
        try:
            # Get key body points
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            left_hip = keypoints[11][:2]
            right_hip = keypoints[12][:2]
            left_knee = keypoints[13][:2]
            right_knee = keypoints[14][:2]
            left_ankle = keypoints[15][:2]
            right_ankle = keypoints[16][:2]
            
            # Calculate average positions
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            knee_y = (left_knee[1] + right_knee[1]) / 2
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            
            # Calculate body proportions
            torso_length = hip_y - shoulder_y
            leg_length = ankle_y - hip_y
            
            # Calculate knee bend (ratio of upper leg to total leg)
            upper_leg = knee_y - hip_y
            lower_leg = ankle_y - knee_y
            
            # Posture classification logic
            if torso_length < 0:  # Inverted (bending over)
                return "Bending"
            
            # Check if legs are bent significantly (sitting)
            if leg_length < torso_length * 0.8:
                return "Sitting"
            
            # Check if torso is significantly angled (bending)
            hip_x = (left_hip[0] + right_hip[0]) / 2
            shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
            torso_angle = abs(hip_x - shoulder_x)
            
            if torso_angle > torso_length * 0.3:
                return "Bending"
            
            # Check knee bend ratio
            if upper_leg < lower_leg * 0.6:
                return "Sitting"
            
            # Default to standing if upright
            return "Standing"
            
        except (IndexError, ValueError):
            return "Unknown"
    
    def draw_results(self, frame, results):
        """
        Draw bounding boxes and posture labels on frame
        
        Args:
            frame: Input image/frame
            results: YOLO detection results
            
        Returns:
            Annotated frame with bounding boxes and labels
        """
        annotated_frame = frame.copy()
        
        for result in results:
            # Draw bounding boxes for all detected objects
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Draw bounding box
                color = (0, 255, 0)  # Green for general objects
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Process pose keypoints if available
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data
                
                for i, kpts in enumerate(keypoints_data):
                    # Classify posture
                    posture = self.classify_posture(kpts.cpu().numpy())
                    
                    # Get bounding box for this person
                    if i < len(boxes):
                        x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
                        
                        # Color code by posture
                        posture_colors = {
                            "Standing": (0, 255, 0),    # Green
                            "Sitting": (255, 165, 0),   # Orange
                            "Bending": (0, 0, 255),     # Red
                            "Unknown": (128, 128, 128)  # Gray
                        }
                        color = posture_colors.get(posture, (255, 255, 255))
                        
                        # Draw posture label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(annotated_frame, f"Posture: {posture}", 
                                   (x1, y2 + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw skeleton
                    self.draw_skeleton(annotated_frame, kpts.cpu().numpy())
        
        return annotated_frame
    
    def draw_skeleton(self, frame, keypoints):
        """
        Draw skeleton connections on the frame
        
        Args:
            frame: Input frame
            keypoints: Array of keypoint coordinates
        """
        # COCO skeleton connections
        skeleton = [
            [5, 6],   # shoulders
            [5, 7],   # left shoulder to elbow
            [7, 9],   # left elbow to wrist
            [6, 8],   # right shoulder to elbow
            [8, 10],  # right elbow to wrist
            [5, 11],  # left shoulder to hip
            [6, 12],  # right shoulder to hip
            [11, 12], # hips
            [11, 13], # left hip to knee
            [13, 15], # left knee to ankle
            [12, 14], # right hip to knee
            [14, 16], # right knee to ankle
        ]
        
        # Draw connections
        for connection in skeleton:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx][:2]
                pt2 = keypoints[pt2_idx][:2]
                
                # Only draw if both points are visible (confidence > 0)
                if keypoints[pt1_idx][2] > 0.5 and keypoints[pt2_idx][2] > 0.5:
                    pt1 = tuple(map(int, pt1))
                    pt2 = tuple(map(int, pt2))
                    cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
        
        # Draw keypoints
        for kpt in keypoints:
            if kpt[2] > 0.5:  # confidence threshold
                x, y = map(int, kpt[:2])
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
    
    def process_image(self, image_path, output_path=None, show=True):
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            show: Whether to display the result
        """
        print(f"\nProcessing image: {image_path}")
        
        # Read image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Run detection
        results = self.model(frame)
        
        # Draw results
        annotated_frame = self.draw_results(frame, results)
        
        # Save output
        if output_path:
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"Saved output to: {output_path}")
        
        # Display
        if show:
            cv2.imshow('YOLO Detection', annotated_frame)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print detection summary
        self.print_detection_summary(results)
    
    def process_video(self, video_path, output_path=None, show=True):
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show: Whether to display the result
        """
        print(f"\nProcessing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame)
            
            # Draw results
            annotated_frame = self.draw_results(frame, results)
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame
            if writer:
                writer.write(annotated_frame)
            
            # Display
            if show:
                cv2.imshow('YOLO Video Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user")
                    break
            
            # Progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            print(f"Saved output video to: {output_path}")
        cv2.destroyAllWindows()
        
        print(f"Completed! Processed {frame_count} frames")
    
    def process_webcam(self):
        """
        Process live webcam feed
        """
        print("\nStarting webcam... Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame)
            
            # Draw results
            annotated_frame = self.draw_results(frame, results)
            
            # Display
            cv2.imshow('YOLO Webcam Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def print_detection_summary(self, results):
        """
        Print summary of detections
        
        Args:
            results: YOLO detection results
        """
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        
        for result in results:
            boxes = result.boxes
            print(f"\nTotal objects detected: {len(boxes)}")
            
            # Count by class
            class_counts = {}
            for box in boxes:
                class_name = result.names[int(box.cls[0])]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("\nDetected objects:")
            for class_name, count in class_counts.items():
                print(f"  - {class_name}: {count}")
            
            # Posture summary
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data
                posture_counts = {}
                
                for kpts in keypoints_data:
                    posture = self.classify_posture(kpts.cpu().numpy())
                    posture_counts[posture] = posture_counts.get(posture, 0) + 1
                
                print("\nPosture classifications:")
                for posture, count in posture_counts.items():
                    print(f"  - {posture}: {count}")
        
        print("="*50 + "\n")


def main():
    """
    Main function to run the detector
    """
    parser = argparse.ArgumentParser(description='YOLO Object and Posture Detection')
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--output', type=str, help='Path to save output')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt', 
                       help='YOLO model to use (default: yolov8n-pose.pt)')
    parser.add_argument('--no-show', action='store_true', help='Do not display output')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PostureDetector(model_name=args.model)
    
    # Process based on input type
    if args.webcam:
        detector.process_webcam()
    elif args.input:
        input_path = Path(args.input)
        
        # Check if input is image or video
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        if input_path.suffix.lower() in image_extensions:
            detector.process_image(input_path, args.output, not args.no_show)
        elif input_path.suffix.lower() in video_extensions:
            detector.process_video(input_path, args.output, not args.no_show)
        else:
            print(f"Error: Unsupported file format {input_path.suffix}")
    else:
        print("Error: Please provide --input or use --webcam")
        parser.print_help()


if __name__ == "__main__":
    main()
