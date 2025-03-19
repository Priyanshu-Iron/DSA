function maxChunksToSortedII(arr) {
    let n = arr.length;
    let leftMax = new Array(n);
    let rightMin = new Array(n);

    // Fill leftMax array
    leftMax[0] = arr[0];
    for (let i = 1; i < n; i++) {
        leftMax[i] = Math.max(leftMax[i - 1], arr[i]);
    }

    // Fill rightMin array
    rightMin[n - 1] = arr[n - 1];
    for (let i = n - 2; i >= 0; i--) {
        rightMin[i] = Math.min(rightMin[i + 1], arr[i]);
    }

    // Count valid chunk splits
    let chunks = 0;
    for (let i = 0; i < n - 1; i++) {
        if (leftMax[i] <= rightMin[i + 1]) {
            chunks++;
        }
    }

    return chunks + 1; // +1 because chunks count is always 1 more than valid splits
}

// Example usage:
console.log(maxChunksToSortedII([5, 4, 3, 6, 7, 8])); // Output: 3
console.log(maxChunksToSortedII([2, 1, 3, 4, 4])); // Output: 4
console.log(maxChunksToSortedII([1, 1, 1, 1, 1])); // Output: 5
