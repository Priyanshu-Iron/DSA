// Check if Array is Sorted and Rotated

let nums = [1, 1, 1];
var isSortedAndRotated = function(nums) {
    let countBreaks = 0;
    let n = nums.length;
    
    for (let i = 0; i < n; i++) {
        if (nums[i] > nums[(i + 1) % n]) {
            countBreaks++;
        }
    }
    
    // If all elements are the same, return true directly
    if (countBreaks === 0) {
        return true;
    }
    
    return countBreaks === 1;
};

console.log(isSortedAndRotated(nums)); // Output will be true
