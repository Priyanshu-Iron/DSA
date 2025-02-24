const nums = [1,2,3,4,5];
const k = 2;

const rotate = (nums, k) => {
    k = k % nums.length;
    const rotatedArray = nums.slice(-k).concat(nums.slice(0, -k));
    return rotatedArray;
}

// console.log(rotate(nums, k));

// In-place solution

const rotateInPlace = (nums, k) => {
    k = k % nums.length;

    let l = 0, r = nums.length - 1;
    while (l < r) {
        [nums[l], nums[r]] = [nums[r], nums[l]];
        l++;
        r--;
    }

    l = 0, r = k - 1;
    while (l < r) {
        [nums[l], nums[r]] = [nums[r], nums[l]];
        l++;
        r--;
    }

    l = k, r = nums.length - 1;
    while (l < r) {
        [nums[l], nums[r]] = [nums[r], nums[l]];
        l++;
        r--;

    }
    return nums;
}

console.log(rotateInPlace(nums, k));