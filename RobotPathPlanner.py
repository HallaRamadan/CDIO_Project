from ultralytics import YOLO
import cv2
import numpy as np
import heapq
import os

# Load the model and image
model = YOLO("models/best.pt")
original_image_path = "Datasats/Images/IMG_3814.JPG"
results = model(source=original_image_path, conf=0.5, show=False, save=True)

# Extract bounding boxes and labels from results
detections = results[0].boxes.data.cpu().numpy()

# Load the original image
original_image = cv2.imread(original_image_path)

# Initialize lists for different objects
ball_coords = []
robot_coords = None
goal_coords = None
obstacle_coords = []
field_size = None

# Class names
class_names = ['Field', 'Goal', 'Orange-ball', 'Robot', 'cross', 'egg', 'white-ball']

# Grid size for pathfinding
GRID_SIZE = (50, 50)

def pixel_to_grid(pixel_coord, field_size):
    return (
        int(pixel_coord[0] * GRID_SIZE[0] / field_size[0]),
        int(pixel_coord[1] * GRID_SIZE[1] / field_size[1])
    )

def grid_to_pixel(grid_coord, field_size):
    return (
        int(grid_coord[0] * field_size[0] / GRID_SIZE[0]),
        int(grid_coord[1] * field_size[1] / GRID_SIZE[1])
    )

for detection in detections:
    xmin, ymin, xmax, ymax, confidence, class_id = detection
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    label = class_names[int(class_id)]
    
    if label in ['white-ball', 'Orange-ball']:
        ball_coords.append((xmin, ymin))
    elif label == 'Robot':
        robot_coords = (xmin, ymin)
    elif label == 'Goal':
        goal_coords = (xmin, ymin)
    elif label == 'Field':
        field_size = (xmax - xmin, ymax - ymin)
    elif label in ['cross', 'egg']:
        obstacle_coords.append((xmin, ymin, xmax, ymax))

if field_size is None:
    raise ValueError("Field not detected in the image")

# Convert pixel coordinates to grid coordinates
grid_ball_coords = [pixel_to_grid(coord, field_size) for coord in ball_coords]
grid_robot_coords = pixel_to_grid(robot_coords, field_size) if robot_coords else None
grid_goal_coords = pixel_to_grid(goal_coords, field_size) if goal_coords else None
grid_obstacle_coords = [
    (
        pixel_to_grid((xmin, ymin), field_size),
        pixel_to_grid((xmax, ymax), field_size)
    ) for xmin, ymin, xmax, ymax in obstacle_coords
]

print("Robot Coordinates:", grid_robot_coords)
print("Ball Coordinates:", grid_ball_coords)
print("Goal Coordinates:", grid_goal_coords)
print("Obstacle Coordinates:", grid_obstacle_coords)

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def astar(start, end, obstacles, grid_size):
    def heuristic(node, end_node):
        return ((node.position[0] - end_node.position[0]) ** 2 + 
                (node.position[1] - end_node.position[1]) ** 2)

    start_node = Node(start)
    end_node = Node(end)
    
    open_list = []
    closed_set = set()
    
    heapq.heappush(open_list, (start_node.f, start_node))
    
    while open_list:
        current_node = heapq.heappop(open_list)[1]
        
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        
        closed_set.add(current_node.position)
        
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares
            node_position = (current_node.position[0] + new_position[0], 
                             current_node.position[1] + new_position[1])
            
            if (node_position[0] < 0 or node_position[0] >= grid_size[0] or 
                node_position[1] < 0 or node_position[1] >= grid_size[1]):
                continue
            
            if any(obs[0][0] <= node_position[0] <= obs[1][0] and 
                   obs[0][1] <= node_position[1] <= obs[1][1] for obs in obstacles):
                continue
            
            if node_position in closed_set:
                continue
            
            new_node = Node(node_position, current_node)
            new_node.g = current_node.g + 1
            new_node.h = heuristic(new_node, end_node)
            new_node.f = new_node.g + new_node.h
            
            if any(open_node.position == new_node.position and new_node.g >= open_node.g 
                   for _, open_node in open_list):
                continue
            
            heapq.heappush(open_list, (new_node.f, new_node))
    
    return None

def plan_path(robot, balls, goal, obstacles, grid_size):
    path = []
    current_position = robot
    ball_collection_order = []
    
    while balls:
        next_ball = min(balls, key=lambda ball: ((ball[0] - current_position[0]) ** 2 + 
                                                 (ball[1] - current_position[1]) ** 2))
        balls.remove(next_ball)
        path_segment = astar(current_position, next_ball, obstacles, grid_size)
        if path_segment:
            path.extend(path_segment[1:])  # Skip the first node to avoid duplication
            ball_collection_order.append(next_ball)
        current_position = next_ball
    
    path_segment = astar(current_position, goal, obstacles, grid_size)
    if path_segment:
        path.extend(path_segment[1:])  # Skip the first node to avoid duplication

    return path, ball_collection_order

def visualize_path(image, path, robot_start, balls, goal, ball_collection_order, field_size, step, obstacle_coords):
    img = image.copy()
    
    # Draw obstacles (crosses)
    for obs in obstacle_coords:
        center = ((obs[0] + obs[2]) // 2, (obs[1] + obs[3]) // 2)
        size = min(obs[2] - obs[0], obs[3] - obs[1]) // 2
        cv2.line(img, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), (0, 0, 255), 2)
        cv2.line(img, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), (0, 0, 255), 2)
    
    # Determine current robot position
    current_position = path[min(step, len(path) - 1)] if step > 0 else robot_start
    
    # Draw balls
    for ball in balls:
        grid_ball = pixel_to_grid(ball, field_size)
        if grid_ball == current_position:
            # Ball is being picked up
            cv2.circle(img, ball, 12, (0, 255, 0), 2)  # Green circle outline
            cv2.circle(img, ball, 5, (0, 255, 0), -1)  # Green dot in center
        elif grid_ball in path[:step]:
            # Ball has been picked up
            cv2.circle(img, ball, 12, (0, 255, 0), 2)  # Green circle outline
            cv2.circle(img, ball, 5, (0, 255, 0), -1)  # Green dot in center
        else:
            # Ball not yet picked up
            cv2.circle(img, ball, 10, (0, 255, 255), -1)  # Yellow circle
    
    # Draw goal
    cv2.circle(img, goal, 15, (0, 0, 255), -1)
    
    # Draw path
    for i in range(1, min(step + 1, len(path))):
        start_point = grid_to_pixel(path[i-1], field_size)
        end_point = grid_to_pixel(path[i], field_size)
        cv2.line(img, start_point, end_point, (0, 255, 0), 2)
    
    # Draw robot
    robot_pos = grid_to_pixel(current_position, field_size)
    cv2.circle(img, robot_pos, 20, (255, 0, 0), -1)
    
    # Add step number to the image
    cv2.putText(img, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def main():
    if grid_robot_coords is None:
        print("Error: Robot not detected in the image")
        return
    if grid_goal_coords is None:
        print("Error: Goal not detected in the image")
        return
    if not grid_ball_coords:
        print("Error: No balls detected in the image")
        return

    path, ball_collection_order = plan_path(grid_robot_coords, grid_ball_coords.copy(), grid_goal_coords, grid_obstacle_coords, GRID_SIZE)
    print("Optimal Path:", path)
    print("Ball Collection Order:", ball_collection_order)

    # Create output directory
    output_dir = "path_visualization"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the path step by step
    for i in range(len(path) + 1):  # +1 to show final state
        print(f"Step {i}:")
        current_position = path[min(i, len(path) - 1)] if i > 0 else grid_robot_coords
        collected_balls = [ball for ball in grid_ball_coords if ball in path[:i]]
        print(f"  Robot position: {current_position}")
        print(f"  Collected balls: {collected_balls}")
        img = visualize_path(original_image, path, robot_coords, ball_coords, goal_coords, ball_collection_order, field_size, i, obstacle_coords)
        cv2.imwrite(os.path.join(output_dir, f'step_{i:03d}.jpg'), img)

    print(f"Path visualization images saved in '{output_dir}' directory")

if __name__ == "__main__":
    main()