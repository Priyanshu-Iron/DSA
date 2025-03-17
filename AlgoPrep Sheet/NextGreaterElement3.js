function nextGreaterElement(n) {
    let digits = n.toString().split('').map(Number);

    // Step 1: Find first decreasing digit from right
    let i = digits.length - 2;
    while (i >= 0 && digits[i] >= digits[i + 1]) {
        i--;
    }

    // If no such index found, return -1 (already largest)
    if (i < 0) return -1;

    // Step 2: Find the smallest digit larger than digits[i] in the right subarray
    let j = digits.length - 1;
    while (digits[j] <= digits[i]) {
        j--;
    }

    // Step 3: Swap digits[i] and digits[j]
    [digits[i], digits[j]] = [digits[j], digits[i]];

    // Step 4: Reverse the digits after index i to get the smallest lexicographical order
    let left = i + 1, right = digits.length - 1;
    while (left < right) {
        [digits[left], digits[right]] = [digits[right], digits[left]];
        left++;
        right--;
    }

    // Step 5: Convert back to integer and check if it's within 32-bit limit
    let result = parseInt(digits.join(''), 10);
    return result > 2 ** 31 - 1 ? -1 : result;
}

// Example usage
console.log(nextGreaterElement(1234)); // Output: 1243
console.log(nextGreaterElement(4321)); // Output: -1
console.log(nextGreaterElement(534976)); // Output: 536479