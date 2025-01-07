// node NameNtimesusingRecursion.js

function Recursion (i,n) {
    if(i>n) return;
    console.log("Priyanshu");
    Recursion(i+1,n)
}

function main () {
    let n = 3
    Recursion(1,n)
}

main()