// node PalindromeNumber.js

let x = 121

function PalindromeNumber (x) {
    let ReverseNumber = 0
    let originalX = x
    let isNegative = x<0

    x = Math.abs(x)

    while(x>0){
        let lastDigit = x%10
        x = Math.floor(x/10)
        ReverseNumber = (ReverseNumber*10)+ lastDigit
    }

    if(ReverseNumber === originalX){
        return true
    }
    else{
        return false
    }
}

console.log(PalindromeNumber(x));