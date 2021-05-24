//
//  FirstVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit
import Alamofire

//=================================
class FirstVC: UIViewController {
    let curs = ["USD","EUR","CHF"]
    // Vert container for all lines
    let vcont = UIView()
    // Hor containers for each line
    let fromCont = UIView()
    let toCont = UIView()
    let btnCont = UIView()
    let resCont = UIView()
    // Widgets
    var fromPicker:AHXPicker!
    var toPicker:AHXPicker!
    let lbRes = UILabel()

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.addSubview( vcont)
    } // viewDidLoad()
    
    //------------------------
    override func layout() {
        self.view.backgroundColor = AHC.bgcol
        // Container for all lines
        AHL.width( vcont, AHC.w * 0.9)
        AHL.height( vcont, AHC.h * 0.33 )
        AHL.subcenter( vcont, self.view)
        AHL.submiddle( vcont, self.view)
        AHL.border( vcont, .blue)
        AHL.border( toCont)
        AHL.border( btnCont)
        

        // Add the line containers
        AHL.vShare( container: vcont, subviews: [fromCont, toCont, btnCont, resCont])

        // From currency
        AHL.border( fromCont, .green)
        let lbFrom = UILabel()
        lbFrom.text = "from:"
        AHL.border( lbFrom)
        let tfFromCur = AHXPickerTf()
        tfFromCur.text = curs[0]
        fromPicker = AHXPicker(tf: tfFromCur, choices: curs) { (idx:Int) in }
        AHL.border( tfFromCur)
        AHL.hShare( container: fromCont, subviews: [lbFrom, tfFromCur])
        
        // To currency
        AHL.border( toCont, .green)
        let lbTo = UILabel()
        lbTo.text = "to:"
        AHL.border( lbTo)
        let tfToCur = AHXPickerTf()
        tfToCur.text = curs[1]
        toPicker = AHXPicker(tf: tfToCur, choices: curs, defaultChoice:1) { (idx:Int) in }
        AHL.border( tfToCur)
        AHL.hShare( container: toCont, subviews: [lbTo, tfToCur])
        
        // Convert button
        let btnConv = UIButton(type: .system)
        AHL.border( btnConv, .magenta)
        btnConv.setTitle( "Convert", for: .normal)
        btnConv.addAction { self.convert() }
        AHL.hShare( container: btnCont, subviews: [btnConv])
        
        // Conversion result. convert() fills this.
        AHL.hShare( container: resCont, subviews: [lbRes])
        convert()
    } // layout()

    /*
    //=============================================
    class ConversionResult: Codable {
        var rate: Double

//        enum CodingKeys: String, CodingKey {
//            case name
//            case email
//            case token
//        }

        public required init(from decoder: Decoder) throws {
            let container = try decoder.unkeyedContainer()
            let tt = 42
//            let container = try decoder.container(keyedBy: CodingKeys.self)
//            self.name = try container.decode(String.self, forKey: .name)
//            self.email = try container.decode(String.self, forKey: .email)
//            self.token = try container.decode(String.self, forKey: .token)
            rate = 1.0
        }
    } // class ConversionResult
    */
    
    func convert() {
        let api_key = "c37b6a0c288bf10ed187"
        let fromCur = "USD" // curs[fromPicker.getChoice()]
        let toCur = "EUR" // curs[toPicker.getChoice()]
        
        AF.request( "https://free.currconv.com/api/v7/convert",
                    method: .get,
                    parameters: ["q":fromCur + "_" + toCur,
                                 "compact":"ultra",
                                 "apiKey":api_key])
            .validate(statusCode: 200..<300)
            .responseJSON { response in
                switch response.result {
                case .success( let JSON):
                    let res = JSON as! [String:Any]
                    var tt = 42
                    var xx:Any = tt
                    var zz = xx as! String
                case .failure(let error):
                     print("Request failed with error: \(error)")
                 }
            }
    }
    

} // class FirstVC
