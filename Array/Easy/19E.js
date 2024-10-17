// 2695. Array Wrapper
/*
Create a class ArrayWrapper that accepts an array of integers in its constructor. This class should have two features:

When two instances of this class are added together with the + operator, the resulting value is the sum of all
the elements in both arrays.
When the String() function is called on the instance, it will return a comma separated string surrounded by brackets.
For example, [1,2,3].

Example 1:
Input: nums = [[1,2],[3,4]], operation = "Add"
Output: 10
Explanation:
const obj1 = new ArrayWrapper([1,2]);
const obj2 = new ArrayWrapper([3,4]);
obj1 + obj2; // 10

Example 2:
Input: nums = [[23,98,42,70]], operation = "String"
Output: "[23,98,42,70]"
Explanation:
const obj = new ArrayWrapper([23,98,42,70]);
String(obj); // "[23,98,42,70]"
*/

class ArrayWrapper {
    constructor(nums) {
        this.nums = nums;
    }

    // Override the default behavior of the + operator
    valueOf() {
        return this.nums.reduce((sum, num) => sum + num, 0);
    }

    // Override the default behavior of String() function
    toString() {
        return `[${this.nums.join(',')}]`;
    }
}

// Example 1:
const obj1 = new ArrayWrapper([1, 2]);
const obj2 = new ArrayWrapper([3, 4]);
console.log(obj1 + obj2);  // Output: 10

// Example 2:
const obj3 = new ArrayWrapper([23, 98, 42, 70]);
console.log(String(obj3));  // Output: "[23,98,42,70]"
