
import Alamofire
import SwiftyJSON

let jstr = """
{"names": ["Bob", "Tim", "Tina"], "ages": [ 20, 30], "weights": [75.1, 80.5, 100] }
"""
let networkData = Data( jstr.utf8)


class myResType {
    class Person {
        var name:String = ""
        var age:Int = 0
        var weight:Double = 0.0
    }
    var arr = [Person]()
    
    
    func fromNetData( _ d:Data?) {
        arr = [Person]()
        if d == nil { return }
        guard let json = try? JSON(data: d!) else { return }
        let weights = json["weights"]
        for (idx,name) in json["names"].enumerated() {
            var p = Person()
            p.name = name.1.stringValue
            p.age = json["ages"][idx].intValue
            p.weight = json["weights"][idx].doubleValue
            arr.append( p)
        }
    } // fromNetData()
} // class MyResType

let res = myResType()
let nild:Data? = nil
res.fromNetData( nild)
print( "NIL")
print( res.arr)

//print("FULL")
//res.fromNetData( networkData)
//print( res.arr)
//print( res.arr[1].name)
//print( res.arr[1].age)
//print( res.arr[1].weight)
//print( res.arr[2].name)
//print( res.arr[2].age)
//print( res.arr[2].weight)

