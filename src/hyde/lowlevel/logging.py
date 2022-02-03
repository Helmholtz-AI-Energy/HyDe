import logging
import sys

__all__ = ["get_logger", "set_basic_config"]


global first_set_of_log_config
first_set_of_log_config = False


def get_logger(level=logging.INFO, log_file=None):
    if first_set_of_log_config:
        return logging.getLogger("HyDe")
    # first_set = glo
    # print(first_set_of_log_config)
    if not first_set_of_log_config:
        set_basic_config(level)
    black, red, green, yellow, blue, magenta, cyan, white = range(8)

    # The background is set with 40 plus the number of the color, and the foreground with 30
    # These are the sequences need to get colored ouput
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    def formatter_message(message, use_color=True):
        if use_color:
            message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
        else:
            message = message.replace("$RESET", "").replace("$BOLD", "")
        return message

    COLORS = {
        "WARNING": magenta,
        "INFO": green,
        "DEBUG": blue,
        "CRITICAL": yellow,
        "ERROR": red,
    }

    class ColoredFormatter(logging.Formatter):
        def __init__(self, msg, use_color=True):
            logging.Formatter.__init__(self, msg, datefmt="%Y-%m-%d %H:%M")
            self.use_color = use_color

        def format(self, record):
            levelname = record.levelname
            if self.use_color and levelname in COLORS:
                levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
                record.levelname = levelname_color
            return logging.Formatter.format(self, record)

    hyde_logger = logging.getLogger("HyDe")
    hyde_stream = logging.StreamHandler()

    hyde_logger.setLevel(level)

    FORMAT = (
        "%(asctime)s - [ $BOLD%(name)-2s$RESET ][ %(levelname)-s ] %(message)s "
        "\t ($BOLD%(filename)s$RESET:%(lineno)d)"
    )
    COLOR_FORMAT = formatter_message(FORMAT, True)

    color_formatter = ColoredFormatter(COLOR_FORMAT)

    hyde_stream.setFormatter(color_formatter)
    hyde_logger.addHandler(hyde_stream)
    hyde_logger.propagate = False

    if log_file is not None:
        fhan = logging.FileHandler(log_file)
        fhan.setLevel(logging.DEBUG)
        hyde_logger.addHandler(fhan)
        # fhan.setFormatter(formatter)
        """comment this to enable requests logger"""
        hyde_logger.disabled = True

    return hyde_logger


def set_basic_config(level):
    global first_set_of_log_config
    first_set_of_log_config = True
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
