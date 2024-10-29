import sys 

def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def to_hms_format(time_str):
    parts = time_str.split()
    return f"{parts[0]}h{parts[1]}m{parts[2]}s"


def to_dms_format(time_str):
    parts = time_str.split()
    return f"{parts[0]}d{parts[1]}m{parts[2]}s"