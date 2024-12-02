// node DigitConcept.js


let n = 7789

while (n>0){
    const lastDigit = n%10;
    n = Math.floor(n/10)
    console.log(lastDigit);
}