//
//  ActionClosure.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-05-22.
//
/*
 Add an action closure to any Control and any event.
 
 Usage:
 
 button.addAction {
     print("Hello, Closure!")
 }
 
 or:
 
 button.addAction(for: .touchUpInside) {
     print("Hello, Closure!")
 }
 
 or if avoiding retain loops:
 
 self.button.addAction(for: .touchUpInside) { [unowned self] in
     self.doStuff()
 }
 */

import UIKit

extension UIControl {
    func addAction(for controlEvents: UIControl.Event = .touchUpInside, _ closure: @escaping()->()) {
        @objc class ClosureSleeve: NSObject {
            let closure:()->()
            init(_ closure: @escaping()->()) { self.closure = closure }
            @objc func invoke() { closure() }
        }
        let sleeve = ClosureSleeve(closure)
        addTarget(sleeve, action: #selector(ClosureSleeve.invoke), for: controlEvents)
        objc_setAssociatedObject(self, "\(UUID())", sleeve, objc_AssociationPolicy.OBJC_ASSOCIATION_RETAIN)
    }
}

