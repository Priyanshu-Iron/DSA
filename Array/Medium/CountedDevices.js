// node CountedDevices.js

let batteryPercentages = [1,1,2,1,3]

function countTestedDevices (){
    let Counted_Devices = 0
    for (let i = 0; i < batteryPercentages.length; i++) {
        if (batteryPercentages[i]>0) {
            Counted_Devices += 1
            for (let j = i+1; j < batteryPercentages.length; j++) {
                batteryPercentages[j] = batteryPercentages[j]-1
            }
        }
    }
    return Counted_Devices
}

console.log(countTestedDevices(batteryPercentages));