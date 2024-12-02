// node ReverseNumber.js

let x = -123

function Reverse () {
    let ReverseNumber = 0;
    let isNegative = x < 0;

    x = Math.abs(x);

    while (x>0) {
        let lastDigit = x%10
        x = Math.floor(x/10)
        ReverseNumber = (ReverseNumber*10)+lastDigit
    }

    if (isNegative) {
        ReverseNumber = -ReverseNumber;
    }

    return ReverseNumber
}

console.log(Reverse(x));