const nums = [1,2,3,1];

function houseRobber(array) {
    let prevMax = 0;
    let currMax = 0;

    for (let i = 0; i < array.length; i++) {
        let temp = currMax;
        currMax = Math.max(prevMax + array[i], currMax);
        prevMax = temp;
    }
    return currMax;
}

console.log(houseRobber(nums))
