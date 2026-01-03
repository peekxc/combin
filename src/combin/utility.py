import numpy as np


def ensure(condition: bool, msg: str):
	if not condition:
		raise ValueError(msg)
