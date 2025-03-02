const nums1 = [[1,2],[2,3],[4,5]]
const nums2 = [[1,4],[3,2],[4,1]]

let m = nums1.length;
let n = nums2.length;

let p1 = 0;
let p2 = 0;
let result = [];

while (p1<m & p2<n) {
    if (nums1[p1][0] < nums2[p2][0]) {
        result.push(nums1[p1]);
        p1++;
    }
    else if (nums1[p1][0] > nums2[p2][0]) {
        result.push(nums2[p2]);
        p2++;
    } else {     
        result.push([nums1[p1][0], nums1[p1][1] + nums2[p2][1]]);
        p1++;
        p2++;
    }
}

while (p1 < m) {
    result.push(nums1[p1]);
    p1++;
}

while (p2 < n) {
    result.push(nums2[p2]);
    p2++;
}

console.log(result);