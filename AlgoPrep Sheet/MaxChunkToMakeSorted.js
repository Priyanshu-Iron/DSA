function maxChunksToSorted(arr) {
    let maxSoFar = 0;
    let chunks = 0;

    for (let i = 0; i < arr.length; i++) {
        maxSoFar = Math.max(maxSoFar, arr[i]);  // Track the max value so far
        if (maxSoFar === i) {  // If maxSoFar matches index, we can make a chunk
            chunks++;
        }
    }

    return chunks;
}

// Example usage:
console.log(maxChunksToSorted([4, 3, 2, 1, 0])); // Output: 1
console.log(maxChunksToSorted([1, 0, 2, 3, 4])); // Output: 4