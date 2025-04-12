import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self):
        self.court_width = 68
        self.court_length = 105

        self.pixel_vertices = None
        self.target_vertices = np.array([
            [0, self.court_width],
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width]
        ]).astype(np.float32)

    def set_pixel_vertices(self, pixel_vertices):
        self.pixel_vertices = np.array(pixel_vertices).astype(np.float32)
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        if self.pixel_vertices is None:
            raise ValueError("Pixel vertices not set. Call set_pixel_vertices() first.")
        
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        if not is_inside:
            return None

        reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed