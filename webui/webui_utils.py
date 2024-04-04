"""
Module containing utility functions for the Gradio WebUI.
Author: Calvin Pang
Date: 2024-03-14
"""

import sys
import base64

#####################################################################################################
#                                                                                                   #
#                                   Terminal Output                                                 #
#                                                                                                   #
#####################################################################################################


class Logger:
    """
    A custom logger class that redirects the output to a file.

    Args:
        filename (str): The name of the file to write the log messages to.

    Attributes:
        terminal (file): The standard output file object.
        log (file): The log file object.

    Methods:
        write(message): Writes the message to both the standard output and the log file.
        flush(): Flushes the output streams.
        isatty(): Returns False.

    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(file=filename, mode="w", encoding="UTF-8")

    def write(self, message):
        """
        Writes the given message to both the terminal and the log file.

        Args:
            message (str): The message to be written.

        Returns:
            None
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Flushes the output of the terminal and the log file.
        """
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        """
        Check if the file descriptor associated with this object is an interactive terminal or not.

        Returns:
            bool: True if the file descriptor is an interactive terminal, False otherwise.
        """
        return False


def read_logs():
    """
    Reads the contents of the 'output.log' file and returns it as a string.

    Returns:
        str: The contents of the 'output.log' file.
    """
    sys.stdout.flush()
    with open(file="webui/output.log", mode="r", encoding="UTF-8") as f:
        return f.read()

#####################################################################################################
#                                                                                                   #
#                                           Branding                                                #
#                                                                                                   #
#####################################################################################################
def image_to_base64(image_path):
    """
    Convert an image file to a base64 encoded string.

    Args:
        image_path: Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def display_logo(image_path):
    """
    Display a logo image in HTML format.

    Args:
        image_path: Path to the image file.

    Returns:
        str: HTML string to display the image.
    """
    string = (
        "<div style=\"text-align:left;\">\n"
    )
    string += f"<img width=\"200\" height=\"200\" src=\"data:image/jpeg;base64,{image_to_base64(image_path)}\" alt=\"ASDAF Logo\">\n"
    string += "</div>"
    return string