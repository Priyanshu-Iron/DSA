// node CountDigits.js

// let num = 121

// let count = 0
// while (num>0) {
//     let LastDigit = num%10
//     if(num%LastDigit === 0){
//         count = count+1
//     }
//     num = Math.floor(num/10)
// }

// console.log(count);

let num = 121;
let count = 0;
let temp = num; // Store the original number

while (temp > 0) {
    let LastDigit = temp % 10; // Get the last digit
    if (LastDigit !== 0 && num % LastDigit === 0) { // Check if num is divisible by LastDigit
        count = count + 1;
    }
    temp = Math.floor(temp / 10); // Remove the last digit
}

console.log(count);