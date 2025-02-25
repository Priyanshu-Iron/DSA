const nums = [-4,-1,0,3,10];
console.log(nums.length);


const sortedSquares = nums => nums.map(num => num * num).sort((a, b) => a - b);

const sortedSquaresTwoPointer = (nums) => {
    let l = 0, r = nums.length - 1;
    let result = new Array(nums.length);
    let index = nums.length - 1;
    
    while (l <= r) {
        if (nums[l] * nums[l] > nums[r] * nums[r]) {
            result[index--] = nums[l] * nums[l];
            l++;
        } else {
            result[index--] = nums[r] * nums[r];
            r--;
        }
    }
    
    return result;
}

console.log(sortedSquaresTwoPointer(nums));