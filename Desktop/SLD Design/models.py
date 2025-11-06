from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class LoadType(Enum):
    MOTOR = "motor"
    HEATER = "heater"
    LIGHTING = "lighting"
    HVAC = "hvac"
    UPS = "ups"
    TRANSFORMER = "transformer"
    CAPACITOR = "capacitor"
    GENERATOR = "generator"
    GENERAL = "general"

class InstallationMethod(Enum):
    CONDUIT = "conduit"
    TRAY = "tray"
    BURIED = "buried"
    AIR = "air"
    DUCT = "duct"
    FREE_AIR = "free_air"

class DutyCycle(Enum):
    CONTINUOUS = "continuous"
    INTERMITTENT = "intermittent"
    SHORT_TIME = "short_time"

class Priority(Enum):
    CRITICAL = "critical"
    ESSENTIAL = "essential"
    NON_ESSENTIAL = "non-essential"

@dataclass
class Load:
    """Internal representation of an electrical load"""
    # Basic identification
    load_id: str
    load_name: str

    # Electrical parameters
    power_kw: float
    voltage: float
    phases: int  # 1 or 3
    power_factor: float = 0.85
    efficiency: float = 0.9

    # Basic identification (continued)
    load_type: LoadType = LoadType.GENERAL

    # Operational parameters
    duty_cycle: DutyCycle = DutyCycle.CONTINUOUS
    starting_method: Optional[str] = None

    # Cable parameters
    cable_length: float = 0.0
    installation_method: InstallationMethod = InstallationMethod.TRAY
    grouping_factor: float = 1.0

    # System parameters
    source_bus: Optional[str] = None
    priority: Priority = Priority.NON_ESSENTIAL
    redundancy: bool = False

    # Additional metadata
    notes: Optional[str] = None

    # Calculated fields (populated by calculation engine)
    current_a: Optional[float] = None
    design_current_a: Optional[float] = None
    apparent_power_kva: Optional[float] = None

    # Cable sizing results
    cable_size_sqmm: Optional[float] = None
    cable_cores: Optional[int] = None
    cable_type: Optional[str] = None
    cable_insulation: Optional[str] = None
    cable_armored: bool = False

    # Protection results
    breaker_rating_a: Optional[float] = None
    breaker_type: Optional[str] = None
    breaker_curve: Optional[str] = None
    fuse_rating_a: Optional[float] = None

    # Voltage drop results
    voltage_drop_v: Optional[float] = None
    voltage_drop_percent: Optional[float] = None

    # Short circuit results
    short_circuit_current_ka: Optional[float] = None
    short_circuit_breaker_ka: Optional[float] = None

    # Power quality
    total_harmonic_distortion: Optional[float] = None
    power_quality_notes: Optional[str] = None

    # Cost estimation
    estimated_cost: Optional[float] = None
    currency: str = "USD"

    def __post_init__(self):
        """Validate data after initialization"""
        if self.power_kw <= 0:
            raise ValueError(f"Power must be positive, got {self.power_kw}")
        if self.voltage not in [230, 400, 415, 440, 690, 3300, 6600, 11000, 33000]:
            raise ValueError(f"Invalid voltage {self.voltage}")
        if self.phases not in [1, 3]:
            raise ValueError(f"Phases must be 1 or 3, got {self.phases}")
        if not (0.1 <= self.power_factor <= 1.0):
            raise ValueError(f"Power factor must be between 0.1 and 1.0, got {self.power_factor}")
        if not (0.1 <= self.efficiency <= 1.0):
            raise ValueError(f"Efficiency must be between 0.1 and 1.0, got {self.efficiency}")
        if self.cable_length < 0.1 or self.cable_length > 1000:
            raise ValueError(f"Cable length must be between 0.1 and 1000 meters, got {self.cable_length}")
        if not (0.3 <= self.grouping_factor <= 1.0):
            raise ValueError(f"Grouping factor must be between 0.3 and 1.0, got {self.grouping_factor}")

@dataclass
class Cable:
    """Internal representation of a cable"""
    cable_id: str
    from_equipment: str
    to_equipment: str

    # Physical properties
    cores: int
    size_sqmm: float
    cable_type: str  # e.g., "XLPE", "PVC", "EPR"
    insulation: str

    # Installation
    length_m: float
    installation_method: InstallationMethod
    armored: bool = False
    shielded: bool = False
    grouping_factor: float = 1.0

    # Electrical properties
    current_carrying_capacity_a: Optional[float] = None
    resistance_ohm_per_km: Optional[float] = None
    reactance_ohm_per_km: Optional[float] = None
    capacitance_uf_per_km: Optional[float] = None

    # Standards compliance
    standard: str = "IEC"
    temperature_rating_c: int = 90

    # Calculated fields
    voltage_drop_v: Optional[float] = None
    voltage_drop_percent: Optional[float] = None
    power_loss_kw: Optional[float] = None
    power_loss_percent: Optional[float] = None

    # Short circuit withstand
    short_circuit_withstand_ka: Optional[float] = None
    short_circuit_duration_s: float = 1.0

    # Cost and procurement
    unit_cost_per_m: Optional[float] = None
    total_cost: Optional[float] = None
    supplier: Optional[str] = None
    part_number: Optional[str] = None

    def get_full_specification(self) -> str:
        """Return full cable specification string"""
        armor = "SWA" if self.armored else ""
        shield = "CTS" if self.shielded else ""
        return f"{self.cores}C x {self.size_sqmm} sq.mm {self.cable_type}/{armor}/{shield}/PVC"

@dataclass
class Breaker:
    """Internal representation of a circuit breaker"""
    breaker_id: str
    load_id: str

    # Basic ratings
    rated_current_a: float
    rated_voltage_v: float
    poles: int

    # Breaking capacity
    breaking_capacity_ka: float

    # Type classification
    type: str  # "MCB", "MCCB", "ACB", "VCB", "SF6"
    breaking_capacity_type: str = "AC"  # AC, DC, or specific
    frame_size: Optional[str] = None

    # Trip characteristics
    curve_type: Optional[str] = None  # "B", "C", "D" for MCBs
    adjustable: bool = False
    thermal_setting_a: Optional[float] = None
    magnetic_setting_a: Optional[float] = None

    # Advanced features
    electronic_trip: bool = False
    earth_leakage_protection: bool = False
    residual_current_ma: Optional[float] = None

    # Standards and compliance
    standard: str = "IEC"
    ip_rating: str = "IP20"
    ics_percent: int = 100  # Service breaking capacity percentage

    # Cost and procurement
    unit_cost: Optional[float] = None
    supplier: Optional[str] = None
    part_number: Optional[str] = None

    def get_standard_rating(self) -> str:
        """Return standard breaker designation"""
        if self.type == "MCB":
            return f"{self.rated_current_a}A {self.curve_type}-curve {self.poles}P"
        elif self.type == "MCCB":
            return f"{self.rated_current_a}A {self.poles}P {self.breaking_capacity_ka}kA"
        else:
            return f"{self.type} {self.rated_current_a}A {self.poles}P"

@dataclass
class Bus:
    """Internal representation of a bus or distribution panel"""
    bus_id: str
    bus_name: str

    # Electrical parameters
    voltage: float
    phases: int

    # Ratings
    rated_current_a: float
    short_circuit_rating_ka: float
    frequency_hz: float = 50
    peak_short_circuit_ka: Optional[float] = None

    # Physical properties
    busbar_material: str = "copper"  # "copper" or "aluminum"
    busbar_configuration: str = "single"  # "single", "double", "triple"

    # System hierarchy
    parent_bus: Optional[str] = None
    child_buses: List[str] = field(default_factory=list)

    # Connected equipment
    connected_loads: List[str] = field(default_factory=list)
    connected_transformers: List[str] = field(default_factory=list)
    connected_generators: List[str] = field(default_factory=list)

    # Loading calculations
    total_load_kw: Optional[float] = None
    total_load_kva: Optional[float] = None
    diversity_factor: float = 1.0  # Default to 1.0 (never 0.000)
    demand_kw: Optional[float] = None
    demand_kva: Optional[float] = None

    # Voltage regulation
    voltage_tolerance_percent: float = 5.0
    actual_voltage_v: Optional[float] = None
    voltage_deviation_percent: Optional[float] = None

    # Protection
    main_breaker_id: Optional[str] = None
    protection_scheme: str = "standard"  # "standard", "differential", "zone"

    # Physical layout
    panel_type: str = "distribution"  # "main", "distribution", "mcc", "control"
    dimensions_mm: Optional[dict] = None  # {"width": x, "height": y, "depth": z}
    location: Optional[str] = None

    def add_load(self, load_id: str):
        """Add a load to this bus"""
        if load_id not in self.connected_loads:
            self.connected_loads.append(load_id)

    def calculate_total_load(self, loads: List[Load]) -> float:
        """Calculate total connected load"""
        total = 0.0
        for load_id in self.connected_loads:
            load = next((l for l in loads if l.load_id == load_id), None)
            if load:
                total += load.power_kw
        self.total_load_kw = total
        return total

@dataclass
class Transformer:
    """Internal representation of a power transformer"""
    transformer_id: str
    name: str

    # Basic ratings
    rating_kva: float
    primary_voltage_v: float
    secondary_voltage_v: float

    # Electrical characteristics
    impedance_percent: float = 6.0
    vector_group: str = "Dyn11"
    no_load_loss_kw: Optional[float] = None
    full_load_loss_kw: Optional[float] = None

    # Construction
    type: str = "oil_immersed"  # "oil_immersed", "dry_type", "cast_resin"
    cooling: str = "ONAN"  # "ONAN", "ONAF", "OFAF", "ODAF"
    windings: str = "copper"  # "copper", "aluminum"

    # Standards and compliance
    standard: str = "IEC"
    frequency_hz: float = 50
    insulation_class: str = "A"  # "A", "E", "B", "F", "H"

    # Operational parameters
    tap_changer: bool = False
    tap_range_percent: float = 0.0
    max_ambient_temp_c: float = 40

    # Calculated fields
    primary_current_a: Optional[float] = None
    secondary_current_a: Optional[float] = None
    efficiency_percent: Optional[float] = None

    # Protection
    buchholz_relay: bool = True
    temperature_relay: bool = True
    pressure_relay: bool = True

    # Cost and procurement
    unit_cost: Optional[float] = None
    supplier: Optional[str] = None
    part_number: Optional[str] = None

    def calculate_currents(self):
        """Calculate primary and secondary currents"""
        self.primary_current_a = (self.rating_kva * 1000) / (self.primary_voltage_v * (3**0.5))
        self.secondary_current_a = (self.rating_kva * 1000) / (self.secondary_voltage_v * (3**0.5))

@dataclass
class Project:
    """Internal representation of an electrical project"""
    project_name: str
    project_id: Optional[str] = None

    # Standards and environment
    standard: str = "IEC"
    voltage_system: str = "LV"
    ambient_temperature_c: float = 40
    altitude_m: float = 0
    soil_resistivity_ohm_m: Optional[float] = None

    # Equipment collections
    loads: List[Load] = field(default_factory=list)
    cables: List[Cable] = field(default_factory=list)
    breakers: List[Breaker] = field(default_factory=list)
    buses: List[Bus] = field(default_factory=list)
    transformers: List[Transformer] = field(default_factory=list)

    # Project metadata
    created_by: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    version: str = "1.0"

    # Calculation results
    total_installed_capacity_kw: Optional[float] = None
    total_demand_kw: Optional[float] = None
    system_diversity_factor: Optional[float] = None
    main_transformer_rating_kva: Optional[float] = None

    # Validation status
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def add_load(self, load: Load):
        """Add a load to the project"""
        self.loads.append(load)

    def get_load_by_id(self, load_id: str) -> Optional[Load]:
        """Get load by ID"""
        return next((l for l in self.loads if l.load_id == load_id), None)

    def validate_project(self) -> tuple[bool, List[str]]:
        """Validate entire project"""
        errors = []
        warnings = []

        # Check for duplicate IDs
        load_ids = [l.load_id for l in self.loads]
        if len(load_ids) != len(set(load_ids)):
            errors.append("Duplicate load IDs found")

        # Check voltage consistency
        voltages = set(l.voltage for l in self.loads)
        if len(voltages) > 2:  # Allow some variation
            warnings.append("Multiple voltage levels detected")

        # Validate cable connections
        # ... additional validation logic

        self.is_valid = len(errors) == 0
        self.validation_errors = errors
        self.validation_warnings = warnings

        return self.is_valid, errors