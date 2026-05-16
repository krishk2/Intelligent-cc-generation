import mediapipe as mp
import numpy as np

class VisualAnalyzer:
    def __init__(self):
        print("Loading MediaPipe Pose and Face Mesh models...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            min_detection_confidence=0.5
        )

    def analyze_sequence_for_reaction(self, frames):
        """
        Calculates a 'reaction score' based on the temporal variance of 
        landmarks across the frame sequence. High variance = sudden movement/reaction.
        """
        if not frames or len(frames) < 2:
            return 0.0
            
        pose_history = []
        face_history = []
        
        for frame_rgb in frames:
            pose_results = self.pose.process(frame_rgb)
            face_results = self.face_mesh.process(frame_rgb)
            
            # Extract nose and shoulders as proxy for sudden body movement
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
                l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                              landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                pose_history.append(nose + l_shoulder + r_shoulder)
                
            # Extract face mesh positions to detect flinching/facial reactions
            if face_results.multi_face_landmarks:
                # Track primary face
                face = face_results.multi_face_landmarks[0].landmark
                # Sample key points (nose tip, chin, left eye)
                sampled_pts = [face[1].x, face[1].y, face[152].x, face[152].y, face[33].x, face[33].y]
                face_history.append(sampled_pts)
                
        # Calculate Reaction Score based on movement variance
        reaction_score = 0.0
        
        if len(pose_history) > 1:
            pose_arr = np.array(pose_history)
            # Calculate variance over time. High variance = rapid movement
            variances = np.var(pose_arr, axis=0)
            pose_score = np.sum(variances) * 500 # Scale up
            reaction_score += pose_score
            
        if len(face_history) > 1:
            face_arr = np.array(face_history)
            variances = np.var(face_arr, axis=0)
            face_score = np.sum(variances) * 1000 # Scale up
            reaction_score += face_score
            
        # Cap score at 1.0
        return min(reaction_score, 1.0)
