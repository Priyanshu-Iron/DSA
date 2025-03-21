function numSubarrayBoundedMax(nums, left, right) {
    let count = 0;
    let subarrayStart = -1;
    let validSubarrays = 0;
    
    for (let i = 0; i < nums.length; i++) {
      if (nums[i] > right) {
        subarrayStart = i;
        validSubarrays = 0;
      }
  
      if (left <= nums[i] && nums[i] <= right) {
        validSubarrays = i - subarrayStart;
      }
  
      count += validSubarrays;
    }
  
    return count;
  }
  
  // Driver code
  const nums1 = [2, 1, 4, 3];
  const left1 = 2;
  const right1 = 3;
  console.log(numSubarrayBoundedMax(nums1, left1, right1));  // Output: 3
  
  const nums2 = [1, 2, 3, 4, 5];
  const left2 = 2;
  const right2 = 4;
  console.log(numSubarrayBoundedMax(nums2, left2, right2));  // Output: 9
  
  const nums3 = [2, 9, 2, 5, 6];
  const left3 = 2;
  const right3 = 8;
  console.log(numSubarrayBoundedMax(nums3, left3, right3));  // Output: 7