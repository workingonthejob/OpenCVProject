import configparser


class IniParser():

    def __init__(self, config):
        self.config_parser = configparser.ConfigParser()
        self.config_file = config
        self.config_parser.read(self.config_file)
        self.ip = self.get_properties('Config', 'CAMERA_URL')
        self.username = self.get_properties('Config', 'USERNAME')
        self.password = self.get_properties('Config', 'PASSWORD')

    def get_properties(self, header, property):
        value = self.config_parser[header][property]
        values = value.split(",")
        return [val.strip() for val in values]

    def save_changes(self):
        with open(self.config_file, 'w') as file:
            self.config_parser.write(file)

    def update_property(self, property, value):
        self.config_parser["Config"][property] = str(value)

    def get_url(self):
        return f'http://{self.ip}/videostream.cgi?user={self.username}&pwd={self.password}'


if __name__ == "__main__":
    pass
