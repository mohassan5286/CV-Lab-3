import cv2
import numpy as np
import matplotlib.pyplot as plt

SAMPLE = 1

def dp_stereo(Il, Ir, sigma=2, c0=1):
    
    assert Il.shape == Ir.shape # Images must have the same size
    H, W = Il.shape
    disparity_left = np.zeros((H, W), dtype=np.float32)
    sigma2 = sigma * sigma

    epsilon = 1e-5  # Small value to handle floating point comparisons
    graph = [[[],[]] for _ in range(H)]
    # print(np.array(graph).shape)

    for row in range(H):
        L = Il[row].astype(np.float32)
        R = Ir[row].astype(np.float32)

        # 1. Cost Matrix (Vectorized)
        # Shape: (W, W). dij[i, j] is cost of matching L[i] and R[j]
        dij = (L[:, None] - R[None, :])**2 / sigma2
        # print(L[:5])
        # print(R[:5])
        # print(L[:5, None] - R[None, :5])

        D = np.zeros((W, W), dtype=np.float32)

        # Lines are matched at both ends (L[0] matches R[0])
        D[0, 0] = dij[0, 0]

        # Initialize boundaries (Accumulate skip costs)
        for i in range(1, W):
            D[i, 0] = D[i-1, 0] + c0  # Skip L[i], keeping R[0] matched
        for j in range(1, W):
            D[0, j] = D[0, j-1] + c0  # Skip R[j], keeping L[0] matched

        # 3. Forward Pass (Fill D)

        for i in range(1, W):
            for j in range(1, W):
                match = D[i-1, j-1] + dij[i, j]
                skip_left = D[i-1, j] + c0
                skip_right = D[i, j-1] + c0

                D[i, j] = min(match, skip_left, skip_right)

        # 4. Backtracking
        i, j = W-1, W-1


        while i > 0 and j > 0:
            current = D[i, j]
            match_cost = D[i-1, j-1] + dij[i, j]
            skip_left_cost = D[i-1, j] + c0
            skip_right_cost = D[i, j-1] + c0
            
            # Check match first
            if abs(current - match_cost) < epsilon:
                # Match found: Record disparity
                disparity_left[row, i] = abs(i - j)
                i -= 1
                j -= 1
            elif abs(current - skip_left_cost) < epsilon:
                # Skip Left (Occlusion in Left Image)
                # Disparity is already 0 from initialization
                i -= 1
            else:
                # Skip Right (Occlusion in Right Image)
                j -= 1
            graph[row][0].append(i)
            graph[row][1].append(j)

        # if row in [9, 128, 256]:
            # plt.figure(figsize=[8, 8])
            # plt.plot(graph[row][0], graph[row][1])
            # plt.title(f'stereo-{SAMPLE}-{row}')
            # plt.savefig(f"stereo-{SAMPLE}-{row}.png")
            # break

    plt.figure(figsize=[8, 8])
    for row in range(H):
        plt.plot(graph[row][0], graph[row][1], alpha=0.5)
    plt.title(f'stereo-{SAMPLE}-rows')
    plt.savefig(f"stereo-{SAMPLE}-rows.png")
    plt.show()
    # plt
        # If i > 0 here, it means we hit j=0. The remaining L pixels are occlusions.
        # Since disparity_left is init with 0s, we don't need to do anything.

    return disparity_left

if __name__ == "__main__":

    # Load grayscale images
    Il = cv2.imread(f"l{SAMPLE}.png", 0)
    Ir = cv2.imread(f"r{SAMPLE}.png", 0)

    if Il is None or Ir is None:
        print("Could not read images. Check filenames.")
        exit()

    print("Computing DP Disparity Map (may take a minute)...")

    disp = dp_stereo(Il, Ir, sigma=2, c0=1)

    # Normalize for saving/showing
    if disp.max() > 0:
        disp_norm = (disp / disp.max() * 255).astype(np.uint8)
    else:
        disp_norm = np.zeros_like(disp, dtype=np.uint8)

    cv2.imwrite("dp_disparity.png", disp_norm)
    print("Saved output: dp_disparity.png")

    cv2.imshow("DP Disparity Map", disp_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
