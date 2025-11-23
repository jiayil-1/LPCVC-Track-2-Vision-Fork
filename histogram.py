import subprocess, os
import matplotlib.pyplot as plt
import numpy as np

with open("durations.txt") as f:
    durations = [float(x.strip()) for x in f if x.strip()]


bins = np.arange(2, 11, 1)

plt.figure()
plt.hist(durations, bins=bins)
plt.xlabel("Video length (seconds)")
plt.ylabel("Number of videos")
plt.title("Histogram of Video Lengths in Train Sample")
plt.savefig("histogram.png")
