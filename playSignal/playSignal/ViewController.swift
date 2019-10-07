//
//  ViewController.swift
//  playSignal
//
//  Created by Andreas Hauenstein on 2019-10-04.
//  Copyright © 2019 AHN. All rights reserved.
//

import Cocoa
import AudioKit

//============================================
class ViewController: NSViewController {

    @IBOutlet var btnRun: NSButton!
    @IBOutlet var txtMain: NSTextView!
    
    let largeLogFont = NSFont( name:"TimesNewRomanPSMT", size:18)
    let smallLogFont = NSFont( name:"TimesNewRomanPSMT", size:12)

    //-----------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
    }

    //-----------------------------------------
    override var representedObject: Any? {
        didSet {
        }
    }

    // Parse a csv string into a list of dicts
    //-------------------------------------------------------------------
    func parseCsv(_ csvstr:String) -> Array<Dictionary<String,String>> {
        let rows = csvstr.components(separatedBy: "\n")
        var header = Array<String>()
        var res = Array<Dictionary<String,String>>()
        var lnum = 0
        for row in rows {
            let trow = row.trimmingCharacters( in: .whitespacesAndNewlines)
            if trow.count == 0 { continue }
            lnum += 1
            if lnum == 1 {
                let cols = row.components(separatedBy: ",")
                header = cols.map( { $0.trimmingCharacters( in: .whitespacesAndNewlines)})
                continue
            }
            var cols = row.components(separatedBy: ",")
            cols = cols.map( { $0.trimmingCharacters( in: .whitespacesAndNewlines)})
            var dict = Dictionary<String,String>()
            for (idx,col) in cols.enumerated() {
                dict[header[idx]] = col
            }
            res.append( dict)
        } // for
        return res
    } // parseCsv()

    // Slurp a text file into a string
    //-----------------------------------------------
    func slurpFile(_ infname:String) -> String {
        do {
            let contents = try String(contentsOfFile: infname)
            return contents
        } catch {
            print("File Read Error: \(error)")
            return ""
        }
    } // slurpFile()

    //----------------------------------------
    @IBAction func btnRun(_ sender: Any) {
        let filestr = slurpFile( "/mac2win/trash/eoec.csv")
        let rows = parseCsv( filestr)
        //playSignal( rows, "pow")
        playSignal( rows, "perc_95_200")
    } // btnRun()

    //------------------------------------------------------------------------------
    func playSignal(_ signal: Array<Dictionary<String,String>>, _ colname: String) {
        struct statics {
            static var firstcall = true
            static var oscillator:AKOscillator!
            static var idx = 0
            static var ttimer = Timer()
            static var mmax = 0.0
        }
        if (statics.firstcall) {
            pr( "starting")
            statics.mmax = 0.0
            statics.firstcall = false
            statics.oscillator = AKOscillator( waveform: AKTable(.sine, count: 4096))
            AudioKit.output = statics.oscillator
            do {
                try AudioKit.start()
            } catch {
                print( "Audiokit failed to start")
            }
            statics.oscillator.rampDuration = 0.0002
            statics.oscillator.frequency = 200
            statics.oscillator.amplitude = 1.0
            statics.oscillator.play()
        }
        statics.ttimer = Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) { timer in
            statics.idx += 1
            statics.idx %= signal.count
            if statics.idx == 0 {
                //stopTimer()
                statics.ttimer.invalidate()
                statics.oscillator.stop()
                statics.firstcall = true
                self.pr( "done")
                do {
                     try AudioKit.stop()
                 } catch {
                     print( "Audiokit failed to stop")
                 }
            }
            var freq = 15_000 * Double( signal[statics.idx][colname]!)!
            statics.mmax *= 0.999
            if freq > statics.mmax {
                statics.mmax = freq
            }
            
            //let val = Double.random( in: 200.0 ..< 1000.0)
            statics.oscillator.frequency = statics.mmax
        }
    } // playSignal
    
    // Print a string to the console, with line num
    //-------------------------------------------------
    func pr(_ msg:String) {
        DispatchQueue.main.async {
            let attributes = [NSAttributedString.Key.foregroundColor: NSColor.white]
            let aStr = NSMutableAttributedString(string:msg+"\n", attributes:attributes)
            self.txtMain.textStorage?.append(aStr)
        }
    } // pr()
}

