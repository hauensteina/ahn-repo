//
//  AHXMain.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

//=====================
class AHXMain {
    static let shared = AHXMain()
    static var app = UIApplication.shared.delegate as! AppDelegate
    static var scene = SceneDelegate.shared!
    var navVC:UINavigationController!
    var VCs = [String:UIViewController]()
    var orderedVCs = [String]()
    
    // Entry Point to App, called from SceneDelegate
    //------------------------------------------------
    func main( navVC:UINavigationController) {
        self.navVC = navVC
        navVC.setNavigationBarHidden( true, animated:false)
        VCs = [
            "FirstVC": FirstVC(),
            "SecondVC": SecondVC()
        ]
        orderedVCs = [ "FirstVC", "SecondVC"]
        pushVC( "FirstVC")
    } // main()
    
    // Push a VC by name
    //-------------------------------
    func pushVC( _ vcName:String) {
        let vc = VCs[vcName]!
        let _ = vc.view // Make sure it's instantiated
        self.navVC.pushViewController( vc, animated: true)
        ContainerVC.shared.bottomVC.selectButton( orderedVCs.firstIndex( of:vcName) ?? 0 )
    } // pushCV()

    // Pop top VC
    //----------------
    func popVC() {
        self.navVC.popViewController(animated: true)
    } // popCV()

    // Replace top VC
    //-------------------------------
    func topVC( _ vcName:String) {
        let vc = VCs[vcName]!
        let _ = vc.view // Make sure it's instantiated
        self.navVC.popViewController( animated:true)
        self.navVC.pushViewController( vc, animated: true)
        ContainerVC.shared.bottomVC.selectButton( orderedVCs.firstIndex( of:vcName) ?? 0 )
    } // topVC()
    
} // class AHXMain
