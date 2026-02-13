//! Recursive descent parser for .neuro files.
//!
//! Consumes tokens from the lexer and produces an AST.

use crate::error::ParseError;
use crate::parser::ast::*;
use crate::parser::lexer::{Lexer, Token};

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    current_span: Span,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(source);
        let (current, current_span) = lexer.next_token()?;
        Ok(Parser {
            lexer,
            current,
            current_span,
        })
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        let (tok, span) = self.lexer.next_token()?;
        self.current = tok;
        self.current_span = span;
        Ok(())
    }

    fn eat_newlines(&mut self) -> Result<(), ParseError> {
        while self.current.is_newline() {
            self.advance()?;
        }
        Ok(())
    }

    fn expect_symbol(&mut self) -> Result<Symbol, ParseError> {
        match &self.current {
            Token::Symbol(s) => {
                let sym = Symbol(s.clone());
                self.advance()?;
                Ok(sym)
            }
            other => Err(ParseError {
                message: format!("expected symbol (e.g. :name), got {other:?}"),
                line: self.current_span.line,
                col: self.current_span.col,
            }),
        }
    }

    fn expect_do(&mut self) -> Result<(), ParseError> {
        match &self.current {
            Token::Do => {
                self.advance()?;
                Ok(())
            }
            other => Err(ParseError {
                message: format!("expected 'do', got {other:?}"),
                line: self.current_span.line,
                col: self.current_span.col,
            }),
        }
    }

    fn expect_end(&mut self, context: &str) -> Result<(), ParseError> {
        self.eat_newlines()?;
        match &self.current {
            Token::End => {
                self.advance()?;
                Ok(())
            }
            other => Err(ParseError {
                message: format!("expected 'end' to close {context}, got {other:?}"),
                line: self.current_span.line,
                col: self.current_span.col,
            }),
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match &self.current {
            Token::Ident(s) => {
                let name = s.clone();
                self.advance()?;
                Ok(name)
            }
            other => Err(ParseError {
                message: format!("expected identifier, got {other:?}"),
                line: self.current_span.line,
                col: self.current_span.col,
            }),
        }
    }

    fn try_ident(&self) -> Option<&str> {
        match &self.current {
            Token::Ident(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn try_bool(&mut self) -> Option<bool> {
        match &self.current {
            Token::Bool(v) => {
                let val = *v;
                let _ = self.advance();
                Some(val)
            }
            _ => None,
        }
    }

    // ─── Top-level ──────────────────────────────────────────────────────────

    pub fn parse_program(&mut self) -> Result<NeuroProgram, ParseError> {
        let mut regions = Vec::new();
        let mut attractors = Vec::new();

        loop {
            self.eat_newlines()?;
            match &self.current {
                Token::Eof => break,
                Token::Region => {
                    regions.push(self.parse_region()?);
                }
                Token::Attractor => {
                    attractors.push(self.parse_attractor()?);
                }
                Token::Body => {
                    // `body :name` is a header line — skip for now (informational)
                    self.advance()?;
                    if matches!(self.current, Token::Symbol(_)) {
                        self.advance()?;
                    }
                }
                other => {
                    return Err(ParseError {
                        message: format!(
                            "expected 'region', 'attractor', or 'body', got {other:?}"
                        ),
                        line: self.current_span.line,
                        col: self.current_span.col,
                    });
                }
            }
        }

        Ok(NeuroProgram {
            regions,
            attractors,
        })
    }

    // ─── Region ─────────────────────────────────────────────────────────────

    fn parse_region(&mut self) -> Result<RegionDef, ParseError> {
        let span = self.current_span;
        self.advance()?; // consume 'region'
        let name = self.expect_symbol()?;
        self.expect_do()?;
        self.eat_newlines()?;

        let mut region = RegionDef {
            name: name.clone(),
            span,
            shape: None,
            proportion: None,
            grows: Vec::new(),
            populate: None,
            differentiations: Vec::new(),
            wiring: Vec::new(),
            gradients: Vec::new(),
            secretes: Vec::new(),
            plasticity_from: Vec::new(),
            consolidation_from: Vec::new(),
            sensitive_to: Vec::new(),
        };

        loop {
            self.eat_newlines()?;
            if matches!(self.current, Token::End | Token::Eof) {
                break;
            }
            self.parse_region_stmt(&mut region)?;
        }

        self.expect_end(&format!("region {name}"))?;
        Ok(region)
    }

    fn parse_region_stmt(&mut self, region: &mut RegionDef) -> Result<(), ParseError> {
        match &self.current {
            Token::Ident(kw) => {
                let keyword = kw.clone();
                match keyword.as_str() {
                    "shape" => {
                        self.advance()?;
                        region.shape = Some(self.expect_symbol()?);
                    }
                    "proportion" => {
                        self.advance()?;
                        region.proportion = Some(self.expect_symbol()?);
                    }
                    "grows" => {
                        region.grows.push(self.parse_grows()?);
                    }
                    "secretes" => {
                        self.advance()?;
                        region.secretes.push(self.expect_symbol()?);
                    }
                    "plasticity_from" => {
                        self.advance()?;
                        region.plasticity_from.push(self.expect_symbol()?);
                    }
                    "consolidation_from" => {
                        self.advance()?;
                        region.consolidation_from.push(self.expect_symbol()?);
                    }
                    "sensitive_to" => {
                        self.advance()?;
                        region.sensitive_to.push(self.expect_symbol()?);
                    }
                    _ => {
                        return Err(ParseError {
                            message: format!(
                                "unknown region statement: '{keyword}'"
                            ),
                            line: self.current_span.line,
                            col: self.current_span.col,
                        });
                    }
                }
            }
            Token::Populate => {
                region.populate = Some(self.parse_populate()?);
            }
            Token::Differentiate => {
                region
                    .differentiations
                    .push(self.parse_differentiate()?);
            }
            Token::Wire => {
                region.wiring.push(self.parse_wire()?);
            }
            Token::Gradient => {
                region.gradients.push(self.parse_gradient()?);
            }
            other => {
                return Err(ParseError {
                    message: format!("unexpected token in region body: {other:?}"),
                    line: self.current_span.line,
                    col: self.current_span.col,
                });
            }
        }
        Ok(())
    }

    // ─── Grows ──────────────────────────────────────────────────────────────

    fn parse_grows(&mut self) -> Result<GrowsDirective, ParseError> {
        let span = self.current_span;
        self.advance()?; // consume 'grows'

        let mode_str = self.expect_ident()?;
        let (mode, target) = match mode_str.as_str() {
            "toward" => (GrowsMode::Toward, Some(self.expect_symbol()?)),
            "facing" => (GrowsMode::Facing, Some(self.expect_symbol()?)),
            "first" => (GrowsMode::First, None),
            other => {
                return Err(ParseError {
                    message: format!(
                        "expected 'toward', 'facing', or 'first' after 'grows', got '{other}'"
                    ),
                    line: span.line,
                    col: span.col,
                });
            }
        };

        Ok(GrowsDirective {
            mode,
            target,
            span,
        })
    }

    // ─── Populate ───────────────────────────────────────────────────────────

    fn parse_populate(&mut self) -> Result<PopulateBlock, ParseError> {
        let span = self.current_span;
        self.advance()?; // consume 'populate'
        self.expect_do()?;
        self.eat_newlines()?;

        let mut entries = Vec::new();
        loop {
            self.eat_newlines()?;
            if matches!(self.current, Token::End) {
                break;
            }
            let entry_span = self.current_span;
            let cell_type = Symbol(self.expect_ident()?);
            let intensity = self.expect_symbol()?;
            entries.push(PopulateEntry {
                cell_type,
                intensity,
                span: entry_span,
            });
        }

        self.expect_end("populate")?;
        Ok(PopulateBlock { entries, span })
    }

    // ─── Differentiate ──────────────────────────────────────────────────────

    fn parse_differentiate(&mut self) -> Result<DifferentiateBlock, ParseError> {
        let span = self.current_span;
        self.advance()?; // consume 'differentiate'
        let target_type = self.expect_symbol()?;
        self.expect_do()?;
        self.eat_newlines()?;

        let mut block = DifferentiateBlock {
            target_type: target_type.clone(),
            span,
            receptive_to: Vec::new(),
            affinity: None,
            abundance: None,
            between: None,
            where_clauses: Vec::new(),
            where_dense: false,
            profile: None,
            inhibitory: None,
            recurrent: None,
            feedforward: None,
            sparse: None,
            period: None,
            bank: None,
            sensitive_to: Vec::new(),
        };

        loop {
            self.eat_newlines()?;
            if matches!(self.current, Token::End) {
                break;
            }
            self.parse_diff_stmt(&mut block)?;
        }

        self.expect_end(&format!("differentiate {target_type}"))?;
        Ok(block)
    }

    fn parse_diff_stmt(&mut self, block: &mut DifferentiateBlock) -> Result<(), ParseError> {
        let keyword = self.expect_ident()?;
        match keyword.as_str() {
            "receptive_to" => {
                block.receptive_to.push(self.expect_symbol()?);
            }
            "affinity" => {
                block.affinity = Some(self.expect_symbol()?);
            }
            "abundance" => {
                block.abundance = Some(self.expect_symbol()?);
            }
            "between" => {
                let a = self.expect_symbol()?;
                self.eat_comma()?;
                let b = self.expect_symbol()?;
                block.between = Some((a, b));
            }
            "where" => {
                // `where :chemical, :level` OR `where_dense` handled separately
                let chem = self.expect_symbol()?;
                self.eat_comma()?;
                let level = self.expect_symbol()?;
                block.where_clauses.push(WhereClause {
                    chemical: chem,
                    level,
                });
            }
            "where_dense" => {
                block.where_dense = true;
            }
            "profile" => {
                block.profile = Some(self.expect_symbol()?);
            }
            "inhibitory" => {
                if let Some(v) = self.try_bool() {
                    block.inhibitory = Some(v);
                } else {
                    block.inhibitory = Some(true);
                }
            }
            "recurrent" => {
                if let Some(v) = self.try_bool() {
                    block.recurrent = Some(v);
                } else {
                    block.recurrent = Some(true);
                }
            }
            "feedforward" => {
                if let Some(v) = self.try_bool() {
                    block.feedforward = Some(v);
                } else {
                    block.feedforward = Some(true);
                }
            }
            "sparse" => {
                if let Some(v) = self.try_bool() {
                    block.sparse = Some(v);
                } else {
                    block.sparse = Some(true);
                }
            }
            "period" => {
                block.period = Some(self.expect_symbol()?);
            }
            "bank" => {
                block.bank = Some(self.expect_symbol()?);
            }
            "sensitive_to" => {
                block.sensitive_to.push(self.expect_symbol()?);
            }
            other => {
                return Err(ParseError {
                    message: format!("unknown differentiate statement: '{other}'"),
                    line: self.current_span.line,
                    col: self.current_span.col,
                });
            }
        }
        Ok(())
    }

    // ─── Wire ───────────────────────────────────────────────────────────────

    fn parse_wire(&mut self) -> Result<WireBlock, ParseError> {
        let span = self.current_span;
        self.advance()?; // consume 'wire'
        let mode = self.expect_symbol()?;
        self.expect_do()?;
        self.eat_newlines()?;

        let mut global_params = WireParams::default();
        let mut zone_overrides = Vec::new();

        loop {
            self.eat_newlines()?;
            if matches!(self.current, Token::End) {
                break;
            }

            if matches!(self.current, Token::Within) {
                zone_overrides.push(self.parse_within()?);
            } else if let Some(kw) = self.try_ident() {
                match kw {
                    "reach" | "fanout" | "strength" => {
                        let key = kw.to_string();
                        self.advance()?;
                        let val = self.expect_symbol()?;
                        match key.as_str() {
                            "reach" => global_params.reach = Some(val),
                            "fanout" => global_params.fanout = Some(val),
                            "strength" => global_params.strength = Some(val),
                            _ => unreachable!(),
                        }
                    }
                    other => {
                        return Err(ParseError {
                            message: format!(
                                "expected 'reach', 'fanout', 'strength', or 'within' in wire block, got '{other}'"
                            ),
                            line: self.current_span.line,
                            col: self.current_span.col,
                        });
                    }
                }
            } else {
                return Err(ParseError {
                    message: format!("unexpected token in wire block: {:?}", self.current),
                    line: self.current_span.line,
                    col: self.current_span.col,
                });
            }
        }

        self.expect_end(&format!("wire {mode}"))?;
        Ok(WireBlock {
            mode,
            span,
            global_params,
            zone_overrides,
        })
    }

    fn parse_within(&mut self) -> Result<WireZoneOverride, ParseError> {
        self.advance()?; // consume 'within'
        let chemical = self.expect_symbol()?;
        self.eat_comma()?;
        let level = self.expect_symbol()?;
        self.expect_do()?;
        self.eat_newlines()?;

        let mut params = WireParams::default();
        loop {
            self.eat_newlines()?;
            if matches!(self.current, Token::End) {
                break;
            }
            let key = self.expect_ident()?;
            let val = self.expect_symbol()?;
            match key.as_str() {
                "reach" => params.reach = Some(val),
                "fanout" => params.fanout = Some(val),
                "strength" => params.strength = Some(val),
                other => {
                    return Err(ParseError {
                        message: format!("unknown wire parameter: '{other}'"),
                        line: self.current_span.line,
                        col: self.current_span.col,
                    });
                }
            }
        }

        self.expect_end("within")?;
        Ok(WireZoneOverride {
            chemical,
            level,
            params,
        })
    }

    // ─── Gradient ───────────────────────────────────────────────────────────

    fn parse_gradient(&mut self) -> Result<GradientBlock, ParseError> {
        let span = self.current_span;
        self.advance()?; // consume 'gradient'
        let name = self.expect_symbol()?;
        self.expect_do()?;
        self.eat_newlines()?;

        let mut block = GradientBlock {
            name: name.clone(),
            span,
            chemical: None,
            source: None,
            diffusion: None,
        };

        loop {
            self.eat_newlines()?;
            if matches!(self.current, Token::End) {
                break;
            }
            let key = self.expect_ident()?;
            let val = self.expect_symbol()?;
            match key.as_str() {
                "chemical" => block.chemical = Some(val),
                "source" => block.source = Some(val),
                "diffusion" => block.diffusion = Some(val),
                other => {
                    return Err(ParseError {
                        message: format!("unknown gradient property: '{other}'"),
                        line: self.current_span.line,
                        col: self.current_span.col,
                    });
                }
            }
        }

        self.expect_end(&format!("gradient {name}"))?;
        Ok(block)
    }

    // ─── Attractor ──────────────────────────────────────────────────────────

    fn parse_attractor(&mut self) -> Result<AttractorDef, ParseError> {
        let span = self.current_span;
        self.advance()?; // consume 'attractor'
        let name = self.expect_symbol()?;
        self.expect_do()?;
        self.eat_newlines()?;

        let mut emits = Vec::new();

        loop {
            self.eat_newlines()?;
            if matches!(self.current, Token::End) {
                break;
            }
            let kw = self.expect_ident()?;
            if kw != "emits" {
                return Err(ParseError {
                    message: format!("expected 'emits' in attractor block, got '{kw}'"),
                    line: self.current_span.line,
                    col: self.current_span.col,
                });
            }
            let emit_span = self.current_span;
            let chemical = self.expect_symbol()?;
            self.eat_comma()?;
            let intensity = self.expect_symbol()?;
            emits.push(EmitsDirective {
                chemical,
                intensity,
                span: emit_span,
            });
        }

        self.expect_end(&format!("attractor {name}"))?;
        Ok(AttractorDef { name, span, emits })
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    fn eat_comma(&mut self) -> Result<(), ParseError> {
        if matches!(self.current, Token::Comma) {
            self.advance()?;
        }
        // Comma is optional — don't error if missing
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> NeuroProgram {
        let mut parser = Parser::new(src).unwrap();
        parser.parse_program().unwrap()
    }

    #[test]
    fn empty_program() {
        let prog = parse("");
        assert!(prog.regions.is_empty());
        assert!(prog.attractors.is_empty());
    }

    #[test]
    fn minimal_region() {
        let prog = parse("region :test do\nend");
        assert_eq!(prog.regions.len(), 1);
        assert_eq!(prog.regions[0].name.as_str(), "test");
    }

    #[test]
    fn region_with_shape() {
        let prog = parse("region :m1 do\n  shape :slab\n  proportion :elongated\nend");
        let r = &prog.regions[0];
        assert_eq!(r.shape.as_ref().unwrap().as_str(), "slab");
        assert_eq!(r.proportion.as_ref().unwrap().as_str(), "elongated");
    }

    #[test]
    fn grows_directives() {
        let prog = parse(
            "region :m1 do\n  grows toward :cortical_trophic\n  grows facing :motor_demand\n  grows first\nend",
        );
        let r = &prog.regions[0];
        assert_eq!(r.grows.len(), 3);
        assert_eq!(r.grows[0].mode, GrowsMode::Toward);
        assert_eq!(r.grows[1].mode, GrowsMode::Facing);
        assert_eq!(r.grows[2].mode, GrowsMode::First);
        assert!(r.grows[2].target.is_none());
    }

    #[test]
    fn populate_block() {
        let prog = parse(
            "region :m1 do\n  populate do\n    pyramidal :dominant\n    inhibitory :substantial\n  end\nend",
        );
        let pop = prog.regions[0].populate.as_ref().unwrap();
        assert_eq!(pop.entries.len(), 2);
        assert_eq!(pop.entries[0].cell_type.as_str(), "pyramidal");
        assert_eq!(pop.entries[0].intensity.as_str(), "dominant");
    }

    #[test]
    fn differentiate_block() {
        let prog = parse(
            "region :m1 do\n  differentiate :sensory do\n    receptive_to :visual_trophic\n    affinity :high\n    abundance :moderate\n  end\nend",
        );
        let d = &prog.regions[0].differentiations[0];
        assert_eq!(d.target_type.as_str(), "sensory");
        assert_eq!(d.receptive_to[0].as_str(), "visual_trophic");
        assert_eq!(d.affinity.as_ref().unwrap().as_str(), "high");
    }

    #[test]
    fn differentiate_with_where_and_inhibitory() {
        let prog = parse(
            "region :m1 do\n  differentiate :gate do\n    where :Wnt, :strong\n    inhibitory true\n    profile :fast_spiking\n  end\nend",
        );
        let d = &prog.regions[0].differentiations[0];
        assert_eq!(d.where_clauses.len(), 1);
        assert_eq!(d.where_clauses[0].chemical.as_str(), "Wnt");
        assert_eq!(d.inhibitory, Some(true));
        assert_eq!(d.profile.as_ref().unwrap().as_str(), "fast_spiking");
    }

    #[test]
    fn wire_with_zones() {
        let prog = parse(
            "region :hippo do\n  wire :proximity do\n    reach :moderate\n    within :BMP, :strong do\n      reach :short\n      fanout :sparse\n    end\n  end\nend",
        );
        let w = &prog.regions[0].wiring[0];
        assert_eq!(w.mode.as_str(), "proximity");
        assert_eq!(w.global_params.reach.as_ref().unwrap().as_str(), "moderate");
        assert_eq!(w.zone_overrides.len(), 1);
        assert_eq!(w.zone_overrides[0].chemical.as_str(), "BMP");
        assert_eq!(
            w.zone_overrides[0].params.reach.as_ref().unwrap().as_str(),
            "short"
        );
    }

    #[test]
    fn gradient_block() {
        let prog = parse(
            "region :hippo do\n  gradient :long_axis do\n    chemical :Wnt\n    source :growth_cone\n    diffusion :standard\n  end\nend",
        );
        let g = &prog.regions[0].gradients[0];
        assert_eq!(g.name.as_str(), "long_axis");
        assert_eq!(g.chemical.as_ref().unwrap().as_str(), "Wnt");
        assert_eq!(g.source.as_ref().unwrap().as_str(), "growth_cone");
    }

    #[test]
    fn secretes_and_plasticity() {
        let prog = parse(
            "region :m1 do\n  secretes :motor_cortex_trophic\n  plasticity_from :use\n  plasticity_from :BDNF\n  consolidation_from :repetition\n  sensitive_to :BDNF\nend",
        );
        let r = &prog.regions[0];
        assert_eq!(r.secretes[0].as_str(), "motor_cortex_trophic");
        assert_eq!(r.plasticity_from.len(), 2);
        assert_eq!(r.consolidation_from[0].as_str(), "repetition");
        assert_eq!(r.sensitive_to[0].as_str(), "BDNF");
    }

    #[test]
    fn attractor_with_emits() {
        let prog = parse(
            "attractor :keyboard do\n  emits :language_trophic, :dominant\n  emits :sensory_trophic, :strong\nend",
        );
        assert_eq!(prog.attractors.len(), 1);
        let a = &prog.attractors[0];
        assert_eq!(a.name.as_str(), "keyboard");
        assert_eq!(a.emits.len(), 2);
        assert_eq!(a.emits[0].chemical.as_str(), "language_trophic");
        assert_eq!(a.emits[0].intensity.as_str(), "dominant");
    }

    #[test]
    fn body_header_ignored() {
        let prog = parse(
            "body :rpi5_desktop\n\nattractor :kb do\n  emits :language_trophic, :strong\nend",
        );
        assert_eq!(prog.attractors.len(), 1);
    }

    #[test]
    fn multi_region_program() {
        let prog = parse(
            "region :brainstem do\n  grows first\n  secretes :reticular_trophic\nend\n\nregion :m1 do\n  grows toward :reticular_trophic\nend",
        );
        assert_eq!(prog.regions.len(), 2);
        assert_eq!(prog.regions[0].name.as_str(), "brainstem");
        assert_eq!(prog.regions[1].name.as_str(), "m1");
    }

    #[test]
    fn comments_everywhere() {
        let prog = parse(
            "# Top comment\nregion :test do\n  # Inside comment\n  shape :slab  # Inline after\nend\n# Bottom",
        );
        assert_eq!(prog.regions.len(), 1);
        assert_eq!(prog.regions[0].shape.as_ref().unwrap().as_str(), "slab");
    }

    #[test]
    fn full_m1_cortex() {
        let src = r#"
# M1 Primary Motor Cortex

region :m1_cortex do
  shape :slab
  proportion :elongated

  grows toward :cortical_trophic
  grows toward :parietal_trophic
  grows facing :motor_demand_trophic

  populate do
    pyramidal   :dominant
    inhibitory  :substantial
    relay       :moderate
    gate        :sparse
    modulator   :rare
  end

  differentiate :sensory do
    receptive_to :premotor_trophic
    affinity :high
    abundance :moderate
  end

  differentiate :motor do
    receptive_to :motor_demand_trophic
    affinity :high
    abundance :moderate
  end

  differentiate :gate do
    between :premotor_trophic, :motor_demand_trophic
    sensitive_to :DA
    profile :fast_spiking
    inhibitory true
    affinity :moderate
    abundance :sparse
  end

  differentiate :relay do
    where_dense
    affinity :selective
    abundance :rare
  end

  wire :proximity do
    reach       :moderate
    fanout      :moderate
    strength    :moderate
  end

  plasticity_from :use
  plasticity_from :BDNF
  consolidation_from :repetition

  secretes :motor_cortex_trophic
end
"#;
        let prog = parse(src);
        let r = &prog.regions[0];
        assert_eq!(r.name.as_str(), "m1_cortex");
        assert_eq!(r.grows.len(), 3);
        assert_eq!(r.populate.as_ref().unwrap().entries.len(), 5);
        assert_eq!(r.differentiations.len(), 4);
        assert_eq!(r.wiring.len(), 1);
        assert_eq!(r.secretes.len(), 1);
        assert_eq!(r.plasticity_from.len(), 2);
    }
}
