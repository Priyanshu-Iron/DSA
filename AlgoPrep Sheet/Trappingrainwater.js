function trapRainWater(height) {
    if (height.length === 0) return 0;

    let left = 0, right = height.length - 1;
    let leftMax = 0, rightMax = 0;
    let trappedWater = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                trappedWater += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                trappedWater += rightMax - height[right];
            }
            right--;
        }
    }

    return trappedWater;
}


function trapRainWater2Method(height) {
    if (height.length === 0) return 0;

    const n = height.length;
    const leftMax = new Array(n).fill(0);
    const rightMax = new Array(n).fill(0);
    let trappedWater = 0;

    // Precompute leftMax for each index
    leftMax[0] = height[0];
    for (let i = 1; i < n; i++) {
        leftMax[i] = Math.max(leftMax[i - 1], height[i]);
    }

    // Precompute rightMax for each index
    rightMax[n - 1] = height[n - 1];
    for (let i = n - 2; i >= 0; i--) {
        rightMax[i] = Math.max(rightMax[i + 1], height[i]);
    }

    // Calculate trapped water
    for (let i = 0; i < n; i++) {
        trappedWater += Math.min(leftMax[i], rightMax[i]) - height[i];
    }

    return trappedWater;
}

// Example usage:
const height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1];
console.log(trapRainWater(height)); // Output: 6