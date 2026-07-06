import os


frequency_low = 1

base_dir = os.path.abspath(os.path.dirname(__file__))
compose_file = os.path.join(base_dir, "../docker/docker-compose.yml")
merged_config_path = os.path.join(base_dir, "tmp", "config.yml")

landing_target_reached_file = 'target_reached.txt'

simulation_status_file = os.path.join(base_dir, "tmp", "simulation_status.txt")

metadata_file_path = os.path.join(base_dir,"metadata_file_path.txt")

ideal_x = -80
ideal_y = 75
ideal_z = 35