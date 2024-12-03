// node Armstrong.js

let num = 6841

let sum = 0
let OriginalNum = num
let digit = num.toString().length

while(num>0){
    let lastDigit = num%10
    sum = sum + Math.pow(lastDigit,digit)
    num = Math.floor(num/10)
}

if(OriginalNum === sum){
    console.log(`The number ${OriginalNum} is a Armstrong Number`);
}
else{
    console.log(`The number ${OriginalNum} is NOT a Armstrong Number`);
}