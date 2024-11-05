/*
If you want to generate permutations in a sequence and get the **next permutation** of an array in lexicographic order, you can use the **"next permutation" algorithm**. This algorithm finds the next lexicographically larger permutation of an array, which can be useful for problems where you need permutations one at a time without generating all of them at once.

Here's how the "next permutation" algorithm works:

### Steps to Find the Next Permutation
1. **Identify the Pivot**: Starting from the end of the array, find the first element that is **smaller than the element immediately after it**. Let's call this element at index `i` the **pivot**.
   
2. **Find the Successor**: Again, starting from the end of the array, find the first element that is **greater than the pivot**. This is called the **successor**.

3. **Swap the Pivot and Successor**: Swap the pivot element with the successor element found in step 2.

4. **Reverse the Suffix**: Reverse the part of the array that comes after the pivot index (i.e., from index `i+1` to the end). This will make the suffix the smallest possible sequence, which is the next lexicographical order.

If no such pivot is found in step 1 (i.e., the array is sorted in descending order), then the current permutation is the largest possible. In this case, reverse the entire array to get the smallest permutation.

### JavaScript Code for the Next Permutation Algorithm

Here's how you can implement this in JavaScript:

```javascript
function nextPermutation(nums) {
    let i = nums.length - 2;

    // Step 1: Find the pivot
    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }

    if (i >= 0) { // If there's a valid pivot
        // Step 2: Find the successor
        let j = nums.length - 1;
        while (nums[j] <= nums[i]) {
            j--;
        }
        // Step 3: Swap the pivot with the successor
        [nums[i], nums[j]] = [nums[j], nums[i]];
    }

    // Step 4: Reverse the suffix
    reverse(nums, i + 1);
}

// Helper function to reverse the array from a given start index to the end
function reverse(nums, start) {
    let end = nums.length - 1;
    while (start < end) {
        [nums[start], nums[end]] = [nums[end], nums[start]];
        start++;
        end--;
    }
}

// Example usage
const nums = [1, 2, 3];
nextPermutation(nums);
console.log(nums); // Output: [1, 3, 2], which is the next permutation after [1, 2, 3]
```

### Explanation of the Example
1. Starting with `nums = [1, 2, 3]`, the algorithm will find that `2` (at index 1) is the pivot because it's the first element from the end that’s smaller than the next element.
2. The successor to `2` is `3` (the last element).
3. The pivot (`2`) is swapped with the successor (`3`), resulting in `[1, 3, 2]`.
4. Since there are no elements after the pivot position to reverse, `[1, 3, 2]` is the final result.

### Notes
- Calling `nextPermutation(nums)` repeatedly will cycle through all permutations in lexicographic order.
- This algorithm has \( O(n) \) time complexity for each call, making it efficient if you only need the next permutation.
*/

function nextPermutation(nums) {
    let i = nums.length - 2;

    // Step 1: Find the pivot
    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }

    if (i >= 0) { // If there's a valid pivot
        // Step 2: Find the successor
        let j = nums.length - 1;
        while (nums[j] <= nums[i]) {
            j--;
        }
        // Step 3: Swap the pivot with the successor
        [nums[i], nums[j]] = [nums[j], nums[i]];
    }

    // Step 4: Reverse the suffix
    reverse(nums, i + 1);
}

// Helper function to reverse the array from a given start index to the end
function reverse(nums, start) {
    let end = nums.length - 1;
    while (start < end) {
        [nums[start], nums[end]] = [nums[end], nums[start]];
        start++;
        end--;
    }
}

// Example usage
const nums = [1, 2, 3];
nextPermutation(nums);
console.log(nums); // Output: [1, 3, 2], which is the next permutation after [1, 2, 3]
