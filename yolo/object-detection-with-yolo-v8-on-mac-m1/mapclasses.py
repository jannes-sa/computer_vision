

class mapclasses:
    # Function to read the file and create the desired map
    def create_map_from_file(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        return {index: line.strip() for index, line in enumerate(lines)}